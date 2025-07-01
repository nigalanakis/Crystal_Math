/* Custom JavaScript for Crystal Structure Analysis Documentation */

document.addEventListener('DOMContentLoaded', function() {
    
    // ========================================================================
    // Add badges to API documentation
    // ========================================================================
    
    function addAPIBadges() {
        // Add GPU badges to functions that use GPU acceleration
        const gpuFunctions = [
            'compute_bond_angles_batch',
            'compute_torsion_angles_batch', 
            'compute_center_of_mass_batch',
            'compute_inertia_tensor_batch',
            'identify_rigid_fragments_batch'
        ];
        
        gpuFunctions.forEach(funcName => {
            const elements = document.querySelectorAll(`[id*="${funcName}"]`);
            elements.forEach(el => {
                if (!el.querySelector('.badge-gpu')) {
                    const badge = document.createElement('span');
                    badge.className = 'badge badge-gpu';
                    badge.textContent = 'GPU Accelerated';
                    badge.title = 'This function uses GPU acceleration for improved performance';
                    
                    const title = el.querySelector('.sig, dt');
                    if (title) {
                        title.appendChild(document.createTextNode(' '));
                        title.appendChild(badge);
                    }
                }
            });
        });
        
        // Add performance badges to performance-critical functions
        const performanceFunctions = [
            'cluster_families',
            '_process_batch',
            'compute_similarity'
        ];
        
        performanceFunctions.forEach(funcName => {
            const elements = document.querySelectorAll(`[id*="${funcName}"]`);
            elements.forEach(el => {
                if (!el.querySelector('.badge-performance')) {
                    const badge = document.createElement('span');
                    badge.className = 'badge badge-performance';
                    badge.textContent = 'Performance Critical';
                    badge.title = 'This function may be performance-critical for large datasets';
                    
                    const title = el.querySelector('.sig, dt');
                    if (title) {
                        title.appendChild(document.createTextNode(' '));
                        title.appendChild(badge);
                    }
                }
            });
        });
    }
    
    // ========================================================================
    // Improve code block functionality
    // ========================================================================
    
    function enhanceCodeBlocks() {
        const codeBlocks = document.querySelectorAll('.highlight');
        
        codeBlocks.forEach(block => {
            // Add language label if available
            const langClass = Array.from(block.classList).find(cls => cls.startsWith('highlight-'));
            if (langClass) {
                const language = langClass.replace('highlight-', '');
                const label = document.createElement('div');
                label.className = 'code-language-label';
                label.textContent = language.toUpperCase();
                label.style.cssText = `
                    position: absolute;
                    top: 0.5em;
                    right: 0.5em;
                    background: rgba(0,0,0,0.1);
                    color: #666;
                    padding: 0.2em 0.5em;
                    border-radius: 3px;
                    font-size: 0.7em;
                    font-weight: bold;
                `;
                
                block.style.position = 'relative';
                block.appendChild(label);
            }
        });
    }
    
    // ========================================================================
    // Add expand/collapse for long parameter lists
    // ========================================================================
    
    function addParameterCollapse() {
        const paramLists = document.querySelectorAll('.field-list');
        
        paramLists.forEach(list => {
            const rows = list.querySelectorAll('tr');
            if (rows.length > 8) { // Only add collapse for long lists
                const toggleBtn = document.createElement('button');
                toggleBtn.textContent = 'Show all parameters';
                toggleBtn.className = 'param-toggle-btn';
                toggleBtn.style.cssText = `
                    background: var(--csa-secondary);
                    color: white;
                    border: none;
                    padding: 0.3em 0.8em;
                    border-radius: 4px;
                    font-size: 0.9em;
                    cursor: pointer;
                    margin: 0.5em 0;
                `;
                
                // Hide parameters beyond the first 5
                for (let i = 5; i < rows.length; i++) {
                    rows[i].style.display = 'none';
                }
                
                let expanded = false;
                toggleBtn.addEventListener('click', function() {
                    expanded = !expanded;
                    for (let i = 5; i < rows.length; i++) {
                        rows[i].style.display = expanded ? 'table-row' : 'none';
                    }
                    toggleBtn.textContent = expanded ? 'Show fewer parameters' : 'Show all parameters';
                });
                
                list.parentNode.insertBefore(toggleBtn, list.nextSibling);
            }
        });
    }
    
    // ========================================================================
    // Add smooth scrolling for anchor links
    // ========================================================================
    
    function addSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        
        anchorLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update URL without jumping
                    history.pushState(null, null, '#' + targetId);
                }
            });
        });
    }
    
    // ========================================================================
    // Add search highlighting in results
    // ========================================================================
    
    function enhanceSearch() {
        // Highlight search terms in results
        const urlParams = new URLSearchParams(window.location.search);
        const searchTerm = urlParams.get('highlight');
        
        if (searchTerm) {
            const content = document.querySelector('.document');
            if (content) {
                highlightText(content, searchTerm);
            }
        }
    }
    
    function highlightText(element, term) {
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        textNodes.forEach(textNode => {
            const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
            if (regex.test(textNode.textContent)) {
                const span = document.createElement('span');
                span.innerHTML = textNode.textContent.replace(regex, '<mark style="background-color: yellow; padding: 0.1em;">$1</mark>');
                textNode.parentNode.replaceChild(span, textNode);
            }
        });
    }
    
    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\    function addSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');');
    }
    
    // ========================================================================
    // Add performance metrics display
    // ========================================================================
    
    function addPerformanceMetrics() {
        // Add performance timing info if available
        if (window.performance && window.performance.timing) {
            const timing = window.performance.timing;
            const loadTime = timing.loadEventEnd - timing.navigationStart;
            
            if (loadTime > 0) {
                const perfDiv = document.createElement('div');
                perfDiv.style.cssText = `
                    position: fixed;
                    bottom: 10px;
                    right: 10px;
                    background: rgba(0,0,0,0.8);
                    color: white;
                    padding: 0.3em 0.6em;
                    border-radius: 4px;
                    font-size: 0.8em;
                    z-index: 1000;
                    opacity: 0.7;
                `;
                perfDiv.textContent = `Page loaded in ${loadTime}ms`;
                
                document.body.appendChild(perfDiv);
                
                // Auto-hide after 3 seconds
                setTimeout(() => {
                    perfDiv.style.opacity = '0';
                    setTimeout(() => perfDiv.remove(), 500);
                }, 3000);
            }
        }
    }
    
    // ========================================================================
    // Add keyboard shortcuts
    // ========================================================================
    
    function addKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + K for search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[name="q"]');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
            
            // Escape to close modals/overlays
            if (e.key === 'Escape') {
                const openOverlays = document.querySelectorAll('.modal, .overlay, .dropdown-open');
                openOverlays.forEach(overlay => {
                    if (overlay.style.display !== 'none') {
                        overlay.style.display = 'none';
                    }
                });
            }
        });
    }
    
    // ========================================================================
    // Add tooltips for technical terms
    // ========================================================================
    
    function addTooltips() {
        const technicalTerms = {
            'vdWFV': 'van der Waals Free Volume - a measure of crystal packing efficiency',
            'HDF5': 'Hierarchical Data Format 5 - a file format for storing large amounts of numerical data',
            'CSD': 'Cambridge Structural Database - the world\'s repository for small-molecule crystal structures',
            'CCDC': 'Cambridge Crystallographic Data Centre - the organization that maintains the CSD',
            'GPU': 'Graphics Processing Unit - used for parallel computation acceleration',
            'PyTorch': 'Open source machine learning framework with GPU acceleration support',
            'Z\'': 'Z-prime - the number of independent molecules in the asymmetric unit',
            'RMSD': 'Root Mean Square Deviation - a measure of similarity between structures'
        };
        
        Object.keys(technicalTerms).forEach(term => {
            const regex = new RegExp(`\\b${escapeRegex(term)}\\b`, 'g');
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                {
                    acceptNode: function(node) {
                        // Skip script, style, and code elements
                        const parent = node.parentElement;
                        if (parent && (parent.tagName === 'SCRIPT' || 
                                     parent.tagName === 'STYLE' || 
                                     parent.tagName === 'CODE' ||
                                     parent.classList.contains('highlight'))) {
                            return NodeFilter.FILTER_REJECT;
                        }
                        return NodeFilter.FILTER_ACCEPT;
                    }
                },
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                if (regex.test(node.textContent)) {
                    textNodes.push(node);
                }
            }
            
            textNodes.forEach(textNode => {
                if (regex.test(textNode.textContent)) {
                    const span = document.createElement('span');
                    span.innerHTML = textNode.textContent.replace(regex, 
                        `<abbr title="${technicalTerms[term]}" style="border-bottom: 1px dotted; cursor: help;">    function addSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');</abbr>`
                    );
                    textNode.parentNode.replaceChild(span, textNode);
                }
            });
        });
    }
    
    // ========================================================================
    // Add copy-to-clipboard for configuration examples
    // ========================================================================
    
    function addConfigCopyButtons() {
        const jsonBlocks = document.querySelectorAll('.highlight-json pre, .highlight-javascript pre');
        
        jsonBlocks.forEach(block => {
            // Check if this looks like a configuration
            const text = block.textContent;
            if (text.includes('"extraction"') || text.includes('"filters"') || text.includes('"actions"')) {
                const copyBtn = document.createElement('button');
                copyBtn.innerHTML = '📋 Copy Config';
                copyBtn.className = 'config-copy-btn';
                copyBtn.style.cssText = `
                    position: absolute;
                    top: 0.5em;
                    left: 0.5em;
                    background: var(--csa-primary);
                    color: white;
                    border: none;
                    padding: 0.3em 0.6em;
                    border-radius: 4px;
                    font-size: 0.8em;
                    cursor: pointer;
                    z-index: 10;
                `;
                
                copyBtn.addEventListener('click', function() {
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.innerHTML = '✅ Copied!';
                        copyBtn.style.background = 'var(--csa-success)';
                        setTimeout(() => {
                            copyBtn.innerHTML = '📋 Copy Config';
                            copyBtn.style.background = 'var(--csa-primary)';
                        }, 2000);
                    });
                });
                
                const container = block.closest('.highlight');
                if (container) {
                    container.style.position = 'relative';
                    container.appendChild(copyBtn);
                }
            }
        });
    }
    
    // ========================================================================
    // Initialize all enhancements
    // ========================================================================
    
    // Run all enhancements
    try {
        addAPIBadges();
        enhanceCodeBlocks();
        addParameterCollapse();
        addSmoothScrolling();
        enhanceSearch();
        addPerformanceMetrics();
        addKeyboardShortcuts();
        addTooltips();
        addConfigCopyButtons();
        
        console.log('CSA Documentation enhancements loaded successfully');
    } catch (error) {
        console.warn('Some CSA documentation enhancements failed to load:', error);
    }
    
    // ========================================================================
    // Add a help overlay for keyboard shortcuts
    // ========================================================================
    
    function createHelpOverlay() {
        const helpButton = document.createElement('button');
        helpButton.innerHTML = '?';
        helpButton.title = 'Keyboard Shortcuts';
        helpButton.style.cssText = `
            position: fixed;
            bottom: 60px;
            right: 20px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--csa-primary);
            color: white;
            border: none;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        const helpOverlay = document.createElement('div');
        helpOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 10000;
            display: none;
            align-items: center;
            justify-content: center;
        `;
        
        helpOverlay.innerHTML = `
            <div style="background: white; padding: 2em; border-radius: 8px; max-width: 400px;">
                <h3 style="margin-top: 0; color: var(--csa-primary);">Keyboard Shortcuts</h3>
                <div style="margin: 1em 0;">
                    <strong>Ctrl/Cmd + K</strong> - Focus search box<br>
                    <strong>Escape</strong> - Close this help<br>
                    <strong>?</strong> - Show this help
                </div>
                <button id="close-help" style="
                    background: var(--csa-primary);
                    color: white;
                    border: none;
                    padding: 0.5em 1em;
                    border-radius: 4px;
                    cursor: pointer;
                ">Close</button>
            </div>
        `;
        
        helpButton.addEventListener('click', () => {
            helpOverlay.style.display = 'flex';
        });
        
        helpOverlay.addEventListener('click', (e) => {
            if (e.target === helpOverlay || e.target.id === 'close-help') {
                helpOverlay.style.display = 'none';
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
                const activeElement = document.activeElement;
                if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    helpOverlay.style.display = 'flex';
                }
            }
        });
        
        document.body.appendChild(helpButton);
        document.body.appendChild(helpOverlay);
    }
    
    // Add help overlay after a short delay
    setTimeout(createHelpOverlay, 1000);
});