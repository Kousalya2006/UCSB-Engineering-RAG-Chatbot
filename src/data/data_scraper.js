const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

class UCSBEngineeringScraper {
    // Sets up scraper configs, initializes data structure
    // Increased concurrent processing to 6 pages for better throughput
    constructor() {
        this.browser = null; // Main Puppeteer browser instance
        this.pages = []; // Pool of pages for concurrent scraping
        this.baseUrl = 'https://catalog.ucsb.edu';
        this.data = { // Main data storage structure
            departments: [],
            programs: [],
            courses: []
        };
        
        // College of Engineering Departments
        this.engineeringDepartments = [
            'bioengineering',
            'chemical-engineering',
            'computer-science',
            'electrical-computer-engineering',
            'materials',
            'mechanical-engineering',
            'media-arts-technology',
            'technology-management'
        ];
        
        // Configuration - increased concurrency for better performance
        this.maxConcurrency = 6; // Increased from 3 to 6
        this.pageTimeout = 30000; // Keep higher timeout for reliability
        this.minDelay = 500; // Slightly more conservative delays
        this.maxDelay = 1500;
        this.retryAttempts = 3; // Retry failed requests
        this.retryDelay = 2000; // Delay between retries
        
        // Progress tracking for incremental saves
        this.completedDepartments = new Set();
        this.outputDir = path.join(__dirname, '..', 'data');
    }

    async init() {
        // Launches browser in non headless mode for captcha, creates pool of pages for concurrent processing
        // Optimizes performance blocks non essential resources
        // Launch browser w/ specific settings
        console.log('Starting UCSB Engineering Catalog Scraper...');
        this.browser = await puppeteer.launch({
            headless: false, // Keep visible for CAPTCHA handling
            defaultViewport: null, // Use full screen
            args: [ // Performance and Compatibility flags
                '--start-maximized',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu',
                '--disable-features=TranslateUI',
                '--no-first-run',
                '--no-default-browser-check'
            ]
        });
        
        // Create a pool of pages for concurrent scraping (increased to 6)
        for (let i = 0; i < this.maxConcurrency; i++) {
            const page = await this.browser.newPage();
            // Sets up request interception to block non essential resources
            await page.setDefaultNavigationTimeout(this.pageTimeout);
            await page.setDefaultTimeout(this.pageTimeout);
            
            // Optimize page settings for speed but keep essential resources
            await page.setRequestInterception(true);
            page.on('request', (req) => {
                // Block only non-essential resources
                if (req.resourceType() === 'font' ||
                    req.resourceType() === 'image' ||
                    req.resourceType() === 'media') {
                    req.abort(); // Block images, fonts, media --> optimizes speed
                } else {
                    req.continue();  // Allows HTML, CSS, JS
                }
            });
            
            this.pages.push(page);
        }
        
        // Ensure output directory exists for incremental saves
        try {
            await fs.mkdir(this.outputDir, { recursive: true });
        } catch (error) {
            // Directory already exists
        }
        
        console.log(`Initialized ${this.pages.length} concurrent pages`);
    }

    async scrapeAllDepartments() {
        // Discovers all departments desired on catalog site
        // Filters for engineering related departments using keywords
        // Processes departments with true concurrency - each tab can move to next department when done
        console.log('Scraping departments...');
        
        // Get department links using first page
        const page = this.pages[0];
        await page.goto(`${this.baseUrl}/departments`);
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        await page.waitForSelector('a[href*="/departments/"]', { timeout: this.pageTimeout });
        
        // Extracts ALL department links
        const departmentLinks = await page.evaluate(() => {
            const links = Array.from(document.querySelectorAll('a[href*="/departments/"]'));
            return links.map(link => ({
                name: link.textContent.trim(),
                url: link.href,
                code: link.href.split('/departments/')[1]?.split('/')[0] || ''
            })).filter(link => link.code && link.code !== 'departments');
        });

        console.log(`Found ${departmentLinks.length} total departments`);
        
        // Filter for engineering departments only
        const engineeringDepartmentLinks = departmentLinks.filter(dept => {
            const deptName = dept.name.toLowerCase();
            const deptCode = dept.code.toLowerCase();
            
            // Desired engineering department keywords
            return deptName.includes('engineering') || 
                   deptName.includes('computer science') ||
                   deptName.includes('materials') ||
                   deptName.includes('technology') ||
                   deptCode.includes('engineering') ||
                   deptCode.includes('computer') ||
                   deptCode.includes('materials') ||
                   deptCode.includes('technology');
        });

        console.log(`\nFound ${engineeringDepartmentLinks.length} engineering departments:`);
        engineeringDepartmentLinks.forEach(dept => {
            console.log(`   - ${dept.name} (${dept.code})`);
        });

        // Process departments with true concurrency - each tab can move to next department when done
        await this.processDepartmentsConcurrently(engineeringDepartmentLinks);
    }

    async processDepartmentsConcurrently(departmentLinks) {
        // True concurrent processing - each page can move to next department when done
        // Implements incremental saving after each department completion
        // Uses a queue-based approach for dynamic work distribution
        
        const departmentQueue = [...departmentLinks];
        const results = [];
        const failed = [];
        let completed = 0;
        
        // Worker function for each concurrent page
        const workerFunction = async (pageIndex) => {
            const page = this.pages[pageIndex];
            
            while (departmentQueue.length > 0) {
                const dept = departmentQueue.shift();
                if (!dept) break; // Queue is empty
                
                let departmentData = null;
                let lastError = null;
                
                // Process department with retry logic
                for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
                    try {
                        console.log(`Page ${pageIndex + 1} processing (attempt ${attempt}/${this.retryAttempts}): ${dept.name} (${dept.code})`);
                        departmentData = await this.scrapeDepartment(dept, pageIndex);
                        
                        if (departmentData) {
                            results.push(departmentData);
                            completed++;
                            
                            // Save department data immediately after completion
                            await this.saveDepartmentData(departmentData);
                            this.completedDepartments.add(dept.code);
                            
                            console.log(`Page ${pageIndex + 1} completed ${dept.name} and saved data (${completed}/${departmentLinks.length})`);
                            break; // Success, move to next department
                        }
                    } catch (error) {
                        lastError = error;
                        console.error(`Page ${pageIndex + 1} attempt ${attempt} failed for ${dept.name}: ${error.message}`);
                        
                        if (attempt < this.retryAttempts) {
                            console.log(`Page ${pageIndex + 1} retrying ${dept.name} in ${this.retryDelay}ms...`);
                            await this.delay(this.retryDelay);
                        }
                    }
                }
                
                if (!departmentData) {
                    // All retries failed - add to failed array
                    console.error(`Page ${pageIndex + 1} failed all ${this.retryAttempts} attempts for ${dept.name}: ${lastError.message}`);
                    failed.push({ item: dept, error: lastError });
                }
                
                // Add delay between departments for this worker
                await this.randomDelay(this.minDelay, this.maxDelay);
            }
        };
        
        // Start all workers concurrently
        const workers = [];
        for (let i = 0; i < this.maxConcurrency; i++) {
            workers.push(workerFunction(i));
        }
        
        // Wait for all workers to complete
        await Promise.all(workers);
        
        // Report any failures
        if (failed.length > 0) {
            console.error(`\nFAILED TO PROCESS ${failed.length} DEPARTMENTS:`);
            failed.forEach(({ item, error }) => {
                console.error(`   - ${item.name} (${item.code}): ${error.message}`);
            });
            
            // Don't throw error - continue with successful departments
            console.log(`\nContinuing with ${results.length} successfully processed departments`);
        }
        
        console.log(`\nSuccessfully processed ${results.length} departments with incremental saves`);
        return results;
    }

    async scrapeDepartment(dept, pageIndex) {
        // Scrapes department overview page, coordinates sequential scraping of programs and courses
        // Sequential not concurrent within department to avoid overwhelming individual department pages

        // Get assigned page from pool
        const page = this.pages[pageIndex];
        
        // Navigate to department overview and scrapes
        const overviewUrl = `${this.baseUrl}/departments/${dept.code}/overview`;
        await page.goto(overviewUrl, { waitUntil: 'domcontentloaded' });
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        // Scrape department overview using multiple selectors
        await page.waitForSelector('body');
        const deptOverview = await page.evaluate(() => {
            const contentSelectors = [
                '.content-area',
                '.main-content',
                '.department-overview',
                'main',
                '.container'
            ];
            
            for (const selector of contentSelectors) {
                const element = document.querySelector(selector);
                if (element) {
                    return element.textContent.trim();
                }
            }
            return document.body.textContent.trim();
        });

        const departmentData = {
            name: dept.name,
            code: dept.code,
            url: overviewUrl,
            overview: deptOverview,
            programs: [],
            courses: []
        };

        // Scrape programs and courses sequentially to avoid overwhelming the server
        console.log(`  Page ${pageIndex + 1} scraping programs for ${dept.name}...`);
        // Sequential processing within department - not concurrent
        const programsData = await this.scrapeDepartmentPrograms(dept.code, page);
        departmentData.programs = programsData;

        console.log(`  Page ${pageIndex + 1} scraping courses for ${dept.name}...`);
        const coursesData = await this.scrapeDepartmentCourses(dept.code, page);
        departmentData.courses = coursesData;

        // Add to main data structure (for final save)
        this.data.departments.push(departmentData);
        this.data.programs.push(...programsData);
        this.data.courses.push(...coursesData);
        
        return departmentData;
    }

    async scrapeDepartmentPrograms(deptCode, page) {
        const programs = [];
        
        const programsUrl = `${this.baseUrl}/departments/${deptCode}/programs`;
        await page.goto(programsUrl, { waitUntil: 'domcontentloaded' });
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        await page.waitForSelector('body');
        
        // Multiselector for finding program links
        const programLinks = await page.evaluate(() => {
            const selectors = [
                'a[href*="/programs/"]',
                'a[href*="/majors/"]',
                'a[href*="/minors/"]',
                '.program-link',
                '.program-item a'
            ];
            
            for (const selector of selectors) {
                const elements = Array.from(document.querySelectorAll(selector));
                if (elements.length > 0) {
                    return elements.map(link => ({
                        name: link.textContent.trim(),
                        url: link.href,
                        code: link.href.split('/programs/')[1] || link.href.split('/majors/')[1] || link.href.split('/minors/')[1] || '' // Extracts code from the URL
                    })).filter(link => link.name && link.url && link.code);
                }
            }
            return [];
        });

        if (programLinks.length > 0) {
            console.log(`  Found ${programLinks.length} programs for ${deptCode}`);
            
            // Process each program individually with retry logic
            for (const programLink of programLinks) {
                let programData = null;
                let lastError = null;
                
                // Individual retry loop per program (max 3)
                for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
                    try {
                        console.log(`    Scraping program: ${programLink.name} (attempt ${attempt}/${this.retryAttempts})`);
                        programData = await this.scrapeIndividualProgram(programLink, page);
                        break; // Success, break retry loop
                    } catch (error) {
                        lastError = error;
                        console.error(`    Attempt ${attempt} failed for program ${programLink.name}: ${error.message}`);
                        
                        if (attempt < this.retryAttempts) {
                            await this.delay(this.retryDelay);
                        }
                    }
                }
                
                if (!programData) {
                    throw new Error(`Failed to scrape program ${programLink.name} after ${this.retryAttempts} attempts: ${lastError.message}`);
                }
                
                programs.push(programData);
                // Delay for respect and to not trigger captcha as much
                await this.randomDelay(this.minDelay, this.maxDelay); 
            }
        } else {
            console.log(`  No programs found for ${deptCode}`);
        }

        return programs;
    }

    async scrapeDepartmentCourses(deptCode, page) {
        const courses = [];
        
        const coursesUrl = `${this.baseUrl}/departments/${deptCode}/courses`;
        await page.goto(coursesUrl, { waitUntil: 'domcontentloaded' });
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        await page.waitForSelector('body');
        
        const courseLinks = await page.evaluate(() => {
            const selectors = [
                'a[href*="/courses/"]',
                '.course-link',
                '.course-item a'
            ];
            
            for (const selector of selectors) {
                const elements = Array.from(document.querySelectorAll(selector));
                if (elements.length > 0) {
                    return elements.map(link => ({
                        name: link.textContent.trim(),
                        url: link.href,
                        code: decodeURIComponent(link.href.split('/courses/')[1] || '')
                    })).filter(link => link.name && link.url && link.code);
                }
            }
            return [];
        });

        if (courseLinks.length > 0) {
            console.log(`  Found ${courseLinks.length} courses for ${deptCode}`);
            
            // Process each course individually with retry logic
            for (const courseLink of courseLinks) {
                let courseData = null;
                let lastError = null;
                
                for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
                    try {
                        console.log(`    Scraping course: ${courseLink.name} (attempt ${attempt}/${this.retryAttempts})`);
                        courseData = await this.scrapeIndividualCourse(courseLink, page);
                        break; // Success, break retry loop
                    } catch (error) {
                        lastError = error;
                        console.error(`    Attempt ${attempt} failed for course ${courseLink.name}: ${error.message}`);
                        
                        if (attempt < this.retryAttempts) {
                            await this.delay(this.retryDelay);
                        }
                    }
                }
                
                if (!courseData) {
                    throw new Error(`Failed to scrape course ${courseLink.name} after ${this.retryAttempts} attempts: ${lastError.message}`);
                }
                
                courses.push(courseData);
                await this.randomDelay(this.minDelay, this.maxDelay);
            }
        } else {
            console.log(`  No courses found for ${deptCode}`);
        }

        return courses;
    }

    async scrapeIndividualProgram(programLink, page) {
        // Extracts detailed program info from individual page
        // Uses hierarchical selector approach
        // Structures into consistent format

        await page.goto(programLink.url, { waitUntil: 'domcontentloaded' });
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        await page.waitForSelector('body');
        
        const programData = await page.evaluate(() => {
            const title = document.querySelector('h1')?.textContent.trim() || 
                         document.querySelector('.program-title')?.textContent.trim() || '';
            
            const contentSelectors = [
                '.program-content',
                '.content-area',
                '.main-content',
                'main',
                '.container'
            ];
            
            let content = '';
            for (const selector of contentSelectors) {
                const element = document.querySelector(selector);
                if (element) {
                    content = element.textContent.trim();
                    break;
                }
            }
            
            const description = document.querySelector('.description')?.textContent.trim() || '';
            const requirements = document.querySelector('.requirements')?.textContent.trim() || '';
            
            return {
                title,
                content: content || document.body.textContent.trim(),
                description,
                requirements
            };
        });

        return {
            name: programLink.name,
            code: programLink.code,
            url: programLink.url,
            title: programData.title,
            content: programData.content,
            description: programData.description,
            requirements: programData.requirements
        };
    }

    async scrapeIndividualCourse(courseLink, page) {
        await page.goto(courseLink.url, { waitUntil: 'domcontentloaded' });
        
        // Check for CAPTCHA and handle it
        if (await this.checkForCaptcha(page)) {
            await this.handleCaptcha(page);
        }
        
        await page.waitForSelector('body');
        
        const courseData = await page.evaluate(() => {
            const title = document.querySelector('h1')?.textContent.trim() || 
                         document.querySelector('.course-title')?.textContent.trim() || '';
            
            const contentSelectors = [
                '.course-content',
                '.content-area',
                '.main-content',
                'main',
                '.container'
            ];
            
            let content = '';
            for (const selector of contentSelectors) {
                const element = document.querySelector(selector);
                if (element) {
                    content = element.textContent.trim();
                    break;
                }
            }
            
            const description = document.querySelector('.description')?.textContent.trim() || '';
            const prerequisites = document.querySelector('.prerequisites')?.textContent.trim() || '';
            const units = document.querySelector('.units')?.textContent.trim() || '';
            
            return {
                title,
                content: content || document.body.textContent.trim(),
                description,
                prerequisites,
                units
            };
        });

        return {
            name: courseLink.name,
            code: courseLink.code,
            url: courseLink.url,
            title: courseData.title,
            content: courseData.content,
            description: courseData.description,
            prerequisites: courseData.prerequisites,
            units: courseData.units
        };
    }

    async saveDepartmentData(departmentData) {
        // Saves individual department data immediately after completion
        // Provides backup in case of crashes and progress tracking
        // Creates timestamped individual department files
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${departmentData.code}_${timestamp}.json`;
        const filePath = path.join(this.outputDir, 'individual_departments', filename);
        
        // Ensure individual departments directory exists
        try {
            await fs.mkdir(path.join(this.outputDir, 'individual_departments'), { recursive: true });
        } catch (error) {
            // Directory already exists
        }
        
        // Save individual department data
        await fs.writeFile(filePath, JSON.stringify(departmentData, null, 2));
        
        // Also update progress file for tracking
        const progressFile = path.join(this.outputDir, 'scraping_progress.json');
        const progressData = {
            timestamp: new Date().toISOString(),
            completedDepartments: Array.from(this.completedDepartments),
            totalDepartments: this.data.departments.length,
            totalPrograms: this.data.programs.length,
            totalCourses: this.data.courses.length,
            lastCompletedDepartment: departmentData.code
        };
        
        await fs.writeFile(progressFile, JSON.stringify(progressData, null, 2));
        
        console.log(`Saved individual department data: ${filename}`);
    }

    async checkForCaptcha(page) {
        // Detects captcha, pauses scraper, waits for user intervention

        try {
            const captchaSelectors = [
                'text=Let\'s confirm you are human',
                'text=Complete the security check',
                'text=verify you are not a bot',
                '[data-callback="onCaptchaComplete"]',
                '.g-recaptcha',
                '#captcha',
                'iframe[src*="recaptcha"]',
                'text=Begin >',
                '.captcha-container',
                '[data-testid="captcha"]'
            ];

            for (const selector of captchaSelectors) {
                try {
                    await page.waitForSelector(selector, { timeout: 1000 });
                    return true; // captcha found
                } catch (e) {
                    continue; // next selector try
                }
            }
            return false;
        } catch (error) {
            return false; // no captcha found
        }
    }

    async handleCaptcha(page) {
        console.log('\nCAPTCHA detected! Pausing scraper...');
        console.log('Please solve the CAPTCHA manually in the browser window');
        console.log('The script will wait for you to complete it');
        console.log('Checking every 5 seconds...');
        
        // Wait for CAPTCHA to be solved
        while (true) {
            await this.delay(5000); // 5 seconds
            
            const stillHasCaptcha = await this.checkForCaptcha(page);
            if (!stillHasCaptcha) {
                console.log('CAPTCHA solved! Continuing scraper...');
                await this.randomDelay(2000, 4000);
                break;
            }
        }
    }

    async saveData() {
        // Creates final output in structured, different formats for different cases
        // Saves complete dataset plus individual collections
        // For RAG chatbot knowledge base ingestion
        
        try {
            await fs.mkdir(this.outputDir, { recursive: true });
        } catch (error) {
            // Directory already exists
        }

        // Save final consolidated data files (4 different JSON files)
        await Promise.all([
            // Complete dataset
            fs.writeFile(
                path.join(this.outputDir, 'ucsb_engineering_catalog.json'),
                JSON.stringify(this.data, null, 2)
            ),
            // Individual types 
            fs.writeFile(
                path.join(this.outputDir, 'ucsb_engineering_departments.json'),
                JSON.stringify(this.data.departments, null, 2)
            ),
            fs.writeFile(
                path.join(this.outputDir, 'ucsb_engineering_programs.json'),
                JSON.stringify(this.data.programs, null, 2)
            ),
            fs.writeFile(
                path.join(this.outputDir, 'ucsb_engineering_courses.json'),
                JSON.stringify(this.data.courses, null, 2)
            )
        ]);

        console.log(`\nFinal engineering data saved to ${this.outputDir}/`);
        console.log(`Final Engineering Summary:`);
        console.log(`   - Departments: ${this.data.departments.length}`);
        console.log(`   - Programs: ${this.data.programs.length}`);
        console.log(`   - Courses: ${this.data.courses.length}`);
        console.log(`   - Individual department files: ${this.completedDepartments.size}`);
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async randomDelay(min = 500, max = 1500) {
        const delay = Math.floor(Math.random() * (max - min + 1)) + min;
        return new Promise(resolve => setTimeout(resolve, delay));
    }

    async close() {
        if (this.browser) {
            await this.browser.close();
        }
    }
}

// Main execution
async function main() {
    const scraper = new UCSBEngineeringScraper();
    
    try {
        await scraper.init();
        await scraper.scrapeAllDepartments();
        await scraper.saveData(); // Final consolidated save
    } catch (error) {
        console.error('Fatal error:', error);
        throw error; // Re-throw to ensure process exits with error
    } finally {
        await scraper.close();
    }
}

// Run the scraper
main().catch(console.error);

module.exports = UCSBEngineeringScraper;