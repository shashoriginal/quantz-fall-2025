// QuantZ Week 1 Orientation - Interactive JavaScript

// Global variables and state management
let currentProblem = null;
let problemHistory = [];
let chartInstances = {};

// Mathematical constants and utilities
const MATH_CONSTANTS = {
    PI: Math.PI,
    E: Math.E,
    SQRT_2PI: Math.sqrt(2 * Math.PI),
    TRADING_DAYS: 252
};

// Utility Functions
class MathUtils {
    static normalCDF(x) {
        // Approximation of cumulative normal distribution
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;
        const p = 0.3275911;
        
        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x) / Math.sqrt(2.0);
        
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        
        return 0.5 * (1.0 + sign * y);
    }
    
    static normalPDF(x) {
        return Math.exp(-0.5 * x * x) / MATH_CONSTANTS.SQRT_2PI;
    }
    
    static randomNormal(mean = 0, std = 1) {
        // Box-Muller transformation
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * std + mean;
    }
    
    static blackScholes(S, K, T, r, sigma, optionType = 'call') {
        const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        const d2 = d1 - sigma * Math.sqrt(T);
        
        if (optionType === 'call') {
            return S * this.normalCDF(d1) - K * Math.exp(-r * T) * this.normalCDF(d2);
        } else {
            return K * Math.exp(-r * T) * this.normalCDF(-d2) - S * this.normalCDF(-d1);
        }
    }
    
    static binomialOptionPrice(S, K, T, r, sigma, n, optionType = 'call') {
        const dt = T / n;
        const u = Math.exp(sigma * Math.sqrt(dt));
        const d = 1 / u;
        const p = (Math.exp(r * dt) - d) / (u - d);
        
        // Initialize option values at expiration
        let optionValues = new Array(n + 1);
        for (let i = 0; i <= n; i++) {
            const ST = S * Math.pow(u, 2 * i - n);
            if (optionType === 'call') {
                optionValues[i] = Math.max(0, ST - K);
            } else {
                optionValues[i] = Math.max(0, K - ST);
            }
        }
        
        // Work backwards through the tree
        for (let j = n - 1; j >= 0; j--) {
            for (let i = 0; i <= j; i++) {
                optionValues[i] = Math.exp(-r * dt) * (p * optionValues[i + 1] + (1 - p) * optionValues[i]);
            }
        }
        
        return optionValues[0];
    }
    
    static calculateGreeks(S, K, T, r, sigma) {
        const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        const d2 = d1 - sigma * Math.sqrt(T);
        
        return {
            delta: this.normalCDF(d1),
            gamma: this.normalPDF(d1) / (S * sigma * Math.sqrt(T)),
            theta: -(S * this.normalPDF(d1) * sigma) / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * this.normalCDF(d2),
            vega: S * this.normalPDF(d1) * Math.sqrt(T),
            rho: K * T * Math.exp(-r * T) * this.normalCDF(d2)
        };
    }
}

// Stochastic Process Simulator
class StochasticProcesses {
    static brownianMotion(n, T, numPaths = 1) {
        const dt = T / n;
        const paths = [];
        
        for (let path = 0; path < numPaths; path++) {
            const values = [0];
            for (let i = 1; i <= n; i++) {
                const dW = MathUtils.randomNormal(0, Math.sqrt(dt));
                values.push(values[i - 1] + dW);
            }
            paths.push(values);
        }
        
        return paths;
    }
    
    static geometricBrownianMotion(S0, mu, sigma, n, T, numPaths = 1) {
        const dt = T / n;
        const paths = [];
        
        for (let path = 0; path < numPaths; path++) {
            const values = [S0];
            for (let i = 1; i <= n; i++) {
                const dW = MathUtils.randomNormal(0, Math.sqrt(dt));
                const St = values[i - 1] * Math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
                values.push(St);
            }
            paths.push(values);
        }
        
        return paths;
    }
    
    static randomWalk(n, numPaths = 1) {
        const paths = [];
        
        for (let path = 0; path < numPaths; path++) {
            const values = [0];
            for (let i = 1; i <= n; i++) {
                const step = Math.random() < 0.5 ? -1 : 1;
                values.push(values[i - 1] + step);
            }
            paths.push(values);
        }
        
        return paths;
    }
    
    static meanRevertingProcess(x0, theta, mu, sigma, n, T, numPaths = 1) {
        const dt = T / n;
        const paths = [];
        
        for (let path = 0; path < numPaths; path++) {
            const values = [x0];
            for (let i = 1; i <= n; i++) {
                const dW = MathUtils.randomNormal(0, Math.sqrt(dt));
                const xt = values[i - 1] + theta * (mu - values[i - 1]) * dt + sigma * dW;
                values.push(xt);
            }
            paths.push(values);
        }
        
        return paths;
    }
}

// Monte Carlo Simulation Engine
class MonteCarloEngine {
    static europeanOption(S, K, T, r, sigma, numSims, optionType = 'call') {
        let payoffSum = 0;
        const payoffs = [];
        
        for (let i = 0; i < numSims; i++) {
            const Z = MathUtils.randomNormal();
            const ST = S * Math.exp((r - 0.5 * sigma * sigma) * T + sigma * Math.sqrt(T) * Z);
            
            let payoff;
            if (optionType === 'call') {
                payoff = Math.max(0, ST - K);
            } else {
                payoff = Math.max(0, K - ST);
            }
            
            payoffs.push(payoff);
            payoffSum += payoff;
        }
        
        const price = Math.exp(-r * T) * (payoffSum / numSims);
        const standardError = Math.sqrt(payoffs.reduce((sum, p) => sum + Math.pow(p - payoffSum / numSims, 2), 0) / (numSims - 1)) / Math.sqrt(numSims);
        
        return {
            price: price,
            standardError: standardError * Math.exp(-r * T),
            payoffs: payoffs
        };
    }
    
    static asianOption(S, K, T, r, sigma, numSims, numSteps, optionType = 'call') {
        let payoffSum = 0;
        const dt = T / numSteps;
        
        for (let i = 0; i < numSims; i++) {
            let St = S;
            let priceSum = S;
            
            for (let j = 1; j <= numSteps; j++) {
                const Z = MathUtils.randomNormal();
                St = St * Math.exp((r - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * Z);
                priceSum += St;
            }
            
            const avgPrice = priceSum / (numSteps + 1);
            let payoff;
            
            if (optionType === 'call') {
                payoff = Math.max(0, avgPrice - K);
            } else {
                payoff = Math.max(0, K - avgPrice);
            }
            
            payoffSum += payoff;
        }
        
        return Math.exp(-r * T) * (payoffSum / numSims);
    }
    
    static barrierOption(S, K, T, r, sigma, barrier, numSims, barrierType = 'up-and-out', optionType = 'call') {
        let payoffSum = 0;
        const numSteps = 252; // Daily monitoring
        const dt = T / numSteps;
        
        for (let i = 0; i < numSims; i++) {
            let St = S;
            let barrierHit = false;
            
            for (let j = 1; j <= numSteps; j++) {
                const Z = MathUtils.randomNormal();
                St = St * Math.exp((r - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * Z);
                
                if (barrierType === 'up-and-out' && St >= barrier) {
                    barrierHit = true;
                    break;
                } else if (barrierType === 'down-and-out' && St <= barrier) {
                    barrierHit = true;
                    break;
                }
            }
            
            let payoff = 0;
            if (!barrierHit) {
                if (optionType === 'call') {
                    payoff = Math.max(0, St - K);
                } else {
                    payoff = Math.max(0, K - St);
                }
            }
            
            payoffSum += payoff;
        }
        
        return Math.exp(-r * T) * (payoffSum / numSims);
    }
}

// Practice Problem Generator
class ProblemGenerator {
    static generateProblem(difficulty, topic) {
        const problems = {
            easy: {
                arbitrage: this.generateArbitrageProblem,
                pricing: this.generateSimplePricingProblem,
                stochastic: this.generateStochasticProblem,
                options: this.generateSimpleOptionProblem,
                greeks: this.generateGreeksProblem
            },
            medium: {
                arbitrage: this.generateComplexArbitrageProblem,
                pricing: this.generateRiskNeutralProblem,
                stochastic: this.generateAdvancedStochasticProblem,
                options: this.generateBinomialProblem,
                greeks: this.generateAdvancedGreeksProblem
            },
            hard: {
                arbitrage: this.generateMultiAssetArbitrageProblem,
                pricing: this.generateComplexPricingProblem,
                stochastic: this.generateMartingaleProblem,
                options: this.generateExoticOptionProblem,
                greeks: this.generatePortfolioGreeksProblem
            }
        };
        
        if (topic === 'all') {
            const topics = Object.keys(problems[difficulty]);
            topic = topics[Math.floor(Math.random() * topics.length)];
        }
        
        return problems[difficulty][topic]();
    }
    
    static generateArbitrageProblem() {
        const priceA = 100 + Math.random() * 20;
        const priceB = priceA + (Math.random() - 0.5) * 5;
        const exchangeRate = 0.8 + Math.random() * 0.4;
        
        const arbitrageProfit = Math.abs(priceA - priceB * exchangeRate);
        
        return {
            question: `Asset A trades at $${priceA.toFixed(2)} in Market 1 and Asset B (identical to A) trades at $${priceB.toFixed(2)} in Market 2. The exchange rate is ${exchangeRate.toFixed(3)}. What is the arbitrage profit per unit?`,
            answer: arbitrageProfit,
            explanation: `Arbitrage profit = |Price_A - Price_B × Exchange_Rate| = |${priceA.toFixed(2)} - ${priceB.toFixed(2)} × ${exchangeRate.toFixed(3)}| = ${arbitrageProfit.toFixed(4)}`,
            topic: 'arbitrage',
            difficulty: 'easy'
        };
    }
    
    static generateSimplePricingProblem() {
        const S = 90 + Math.random() * 20;
        const K = 95 + Math.random() * 10;
        const T = 0.1 + Math.random() * 0.4;
        const r = 0.02 + Math.random() * 0.08;
        
        const intrinsicValue = Math.max(0, S - K);
        const timeValue = K * Math.exp(-r * T) - K;
        
        return {
            question: `A call option has a current stock price of $${S.toFixed(2)}, strike price of $${K.toFixed(2)}, time to expiry of ${T.toFixed(2)} years, and risk-free rate of ${(r * 100).toFixed(1)}%. What is the intrinsic value?`,
            answer: intrinsicValue,
            explanation: `Intrinsic value = max(0, S - K) = max(0, ${S.toFixed(2)} - ${K.toFixed(2)}) = ${intrinsicValue.toFixed(4)}`,
            topic: 'pricing',
            difficulty: 'easy'
        };
    }
    
    static generateStochasticProblem() {
        const sigma = 0.15 + Math.random() * 0.25;
        const T = 0.5 + Math.random() * 1.5;
        
        const variance = sigma * sigma * T;
        
        return {
            question: `A stock follows geometric Brownian motion with volatility ${(sigma * 100).toFixed(1)}%. What is the variance of log returns over ${T.toFixed(2)} years?`,
            answer: variance,
            explanation: `Variance of log returns = σ² × T = ${sigma.toFixed(3)}² × ${T.toFixed(2)} = ${variance.toFixed(6)}`,
            topic: 'stochastic',
            difficulty: 'easy'
        };
    }
    
    static generateSimpleOptionProblem() {
        const S = 95 + Math.random() * 10;
        const K = 100;
        const T = 0.25;
        const r = 0.05;
        const sigma = 0.2;
        
        const price = MathUtils.blackScholes(S, K, T, r, sigma, 'call');
        
        return {
            question: `Calculate the Black-Scholes price of a European call option with S=$${S.toFixed(2)}, K=$${K}, T=${T} years, r=${(r * 100)}%, σ=${(sigma * 100)}%.`,
            answer: price,
            explanation: `Using Black-Scholes formula with given parameters: ${price.toFixed(4)}`,
            topic: 'options',
            difficulty: 'easy'
        };
    }
    
    static generateGreeksProblem() {
        const S = 100;
        const K = 100;
        const T = 0.25;
        const r = 0.05;
        const sigma = 0.2;
        
        const greeks = MathUtils.calculateGreeks(S, K, T, r, sigma);
        
        return {
            question: `For an at-the-money call option (S=K=$100, T=0.25 years, r=5%, σ=20%), what is the delta?`,
            answer: greeks.delta,
            explanation: `Delta measures price sensitivity to underlying changes. For ATM call: ${greeks.delta.toFixed(4)}`,
            topic: 'greeks',
            difficulty: 'easy'
        };
    }
    
    // Additional problem generators for medium and hard difficulties would be implemented here
    static generateComplexArbitrageProblem() {
        // Implementation for complex arbitrage problems
        return this.generateArbitrageProblem(); // Placeholder
    }
    
    static generateRiskNeutralProblem() {
        // Implementation for risk-neutral pricing problems
        return this.generateSimplePricingProblem(); // Placeholder
    }
    
    // ... other problem generators
}

// Event Listeners and DOM Manipulation
class UIController {
    static init() {
        this.setupEventListeners();
        this.renderFormulas();
        this.initializePlots();
    }
    
    static setupEventListeners() {
        // Arbitrage Detection
        const detectArbitrageBtn = document.getElementById('detect-arbitrage');
        if (detectArbitrageBtn) {
            detectArbitrageBtn.addEventListener('click', () => this.detectArbitrage());
        }
        
        // Risk-Neutral Pricing
        const calculatePriceBtn = document.getElementById('calculate-price');
        if (calculatePriceBtn) {
            calculatePriceBtn.addEventListener('click', () => this.calculateRiskNeutralPrice());
        }
        
        // Stochastic Process Simulation
        const simulateProcessBtn = document.getElementById('simulate-process');
        if (simulateProcessBtn) {
            simulateProcessBtn.addEventListener('click', () => this.simulateStochasticProcess());
        }
        
        // Binomial Tree
        const generateTreeBtn = document.getElementById('generate-tree');
        if (generateTreeBtn) {
            generateTreeBtn.addEventListener('click', () => this.generateBinomialTree());
        }
        
        // Option Pricing
        const priceOptionBtn = document.getElementById('price-option');
        if (priceOptionBtn) {
            priceOptionBtn.addEventListener('click', () => this.priceOption());
        }
        
        // Greeks and Hedging
        const simulateHedgingBtn = document.getElementById('simulate-hedging');
        if (simulateHedgingBtn) {
            simulateHedgingBtn.addEventListener('click', () => this.simulateHedging());
        }
        
        // Monte Carlo
        const runMonteCarloBtn = document.getElementById('run-monte-carlo');
        if (runMonteCarloBtn) {
            runMonteCarloBtn.addEventListener('click', () => this.runMonteCarlo());
        }
        
        // Market Microstructure
        const simulateOrderBtn = document.getElementById('simulate-order');
        if (simulateOrderBtn) {
            simulateOrderBtn.addEventListener('click', () => this.simulateOrderBook());
        }
        
        // Case Studies
        document.querySelectorAll('.case-study-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.showCaseStudy(e.target.dataset.case));
        });
        
        // Practice Problems
        const generateProblemBtn = document.getElementById('generate-problem');
        if (generateProblemBtn) {
            generateProblemBtn.addEventListener('click', () => this.generateNewProblem());
        }
        
        const checkAnswerBtn = document.getElementById('check-answer');
        if (checkAnswerBtn) {
            checkAnswerBtn.addEventListener('click', () => this.checkAnswer());
        }
    }
    
    static renderFormulas() {
        // Risk-neutral pricing formula
        if (typeof katex !== 'undefined') {
            const formulas = {
                'risk-neutral-formula': 'V_0 = e^{-rT} \\mathbb{E}^Q[V_T]',
                'brownian-formula': 'dW_t \\sim N(0, dt)',
                'martingale-formula': '\\mathbb{E}[X_{t+s} | \\mathcal{F}_t] = X_t',
                'random-walk-formula': 'X_{n+1} = X_n + \\epsilon_{n+1}',
                'delta-formula': '\\Delta = \\frac{\\partial V}{\\partial S}',
                'gamma-formula': '\\Gamma = \\frac{\\partial^2 V}{\\partial S^2}',
                'theta-formula': '\\Theta = \\frac{\\partial V}{\\partial t}',
                'vega-formula': '\\nu = \\frac{\\partial V}{\\partial \\sigma}',
                'black-scholes-pde': '\\frac{\\partial V}{\\partial t} + \\frac{1}{2}\\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} + rS\\frac{\\partial V}{\\partial S} - rV = 0'
            };
            
            Object.entries(formulas).forEach(([id, formula]) => {
                const element = document.getElementById(id);
                if (element) {
                    katex.render(formula, element, {
                        throwOnError: false,
                        displayMode: true
                    });
                }
            });
        }
    }
    
    static initializePlots() {
        // Initialize empty plot containers
        const plotContainers = [
            'stochastic-plot',
            'binomial-tree',
            'hedging-plot',
            'monte-carlo-plot',
            'orderbook-plot',
            'case-study-simulation'
        ];
        
        plotContainers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #888;">Click the button above to generate visualization</div>';
            }
        });
    }
    
    static detectArbitrage() {
        const priceA = parseFloat(document.getElementById('asset-price-a').value);
        const priceB = parseFloat(document.getElementById('asset-price-b').value);
        const exchangeRate = parseFloat(document.getElementById('exchange-rate').value);
        
        const adjustedPriceB = priceB * exchangeRate;
        const arbitrageOpportunity = Math.abs(priceA - adjustedPriceB);
        const threshold = 0.01; // 1 cent threshold
        
        const resultDiv = document.getElementById('arbitrage-result');
        
        if (arbitrageOpportunity > threshold) {
            const strategy = priceA > adjustedPriceB ? 
                `Buy Asset B at $${priceB.toFixed(2)}, sell Asset A at $${priceA.toFixed(2)}` :
                `Buy Asset A at $${priceA.toFixed(2)}, sell Asset B at $${priceB.toFixed(2)}`;
            
            resultDiv.innerHTML = `
                <div class="text-success">
                    <h4>🚨 Arbitrage Opportunity Detected!</h4>
                    <p><strong>Profit per unit:</strong> $${arbitrageOpportunity.toFixed(4)}</p>
                    <p><strong>Strategy:</strong> ${strategy}</p>
                    <p><strong>Adjusted Price B:</strong> $${adjustedPriceB.toFixed(4)}</p>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="text-success">
                    <h4>✅ No Arbitrage</h4>
                    <p>Price difference: $${arbitrageOpportunity.toFixed(4)} (below threshold)</p>
                    <p>Market appears efficient - no risk-free profit opportunities.</p>
                </div>
            `;
        }
    }
    
    static calculateRiskNeutralPrice() {
        const S = parseFloat(document.getElementById('spot-price').value);
        const K = parseFloat(document.getElementById('strike-price').value);
        const r = parseFloat(document.getElementById('risk-free-rate').value) / 100;
        const T = parseFloat(document.getElementById('time-to-expiry').value);
        
        // Simple binomial model for demonstration
        const sigma = 0.2; // Assumed volatility
        const n = 100; // Number of steps
        
        const callPrice = MathUtils.binomialOptionPrice(S, K, T, r, sigma, n, 'call');
        const putPrice = MathUtils.binomialOptionPrice(S, K, T, r, sigma, n, 'put');
        
        const resultDiv = document.getElementById('pricing-result');
        resultDiv.innerHTML = `
            <div>
                <h4>Risk-Neutral Option Prices</h4>
                <p><strong>Call Option:</strong> $${callPrice.toFixed(4)}</p>
                <p><strong>Put Option:</strong> $${putPrice.toFixed(4)}</p>
                <p><strong>Put-Call Parity Check:</strong> ${(callPrice - putPrice + K * Math.exp(-r * T) - S).toFixed(6)}</p>
                <p class="text-muted">Using binomial model with σ=20%, n=100 steps</p>
            </div>
        `;
    }
    
    static simulateStochasticProcess() {
        try {
            const processTypeElement = document.getElementById('process-type');
            const numPathsElement = document.getElementById('num-paths');
            const timeStepsElement = document.getElementById('time-steps');
            const volatilityElement = document.getElementById('volatility');
            
            const processType = processTypeElement ? processTypeElement.value : 'brownian';
            const numPaths = numPathsElement ? parseInt(numPathsElement.value) || 5 : 5;
            const timeSteps = timeStepsElement ? parseInt(timeStepsElement.value) || 100 : 100;
            const volatility = volatilityElement ? parseFloat(volatilityElement.value) || 0.2 : 0.2;
            
            console.log(`Simulating ${processType} with ${numPaths} paths, ${timeSteps} steps, volatility ${volatility}`);
            
            let paths;
            const T = 1; // 1 year
            
            switch (processType) {
                case 'brownian':
                    paths = StochasticProcesses.brownianMotion(timeSteps, T, numPaths);
                    break;
                case 'geometric':
                    paths = StochasticProcesses.geometricBrownianMotion(100, 0.05, volatility, timeSteps, T, numPaths);
                    break;
                case 'random-walk':
                    paths = StochasticProcesses.randomWalk(timeSteps, numPaths);
                    break;
            case 'mean-reverting':
                paths = StochasticProcesses.meanRevertingProcess(100, 0.5, 100, volatility * 10, timeSteps, T, numPaths);
                break;
            default:
                paths = StochasticProcesses.brownianMotion(timeSteps, T, numPaths);
        }
        
        this.plotStochasticPaths(paths, processType);
        } catch (error) {
            console.error('Error in simulateStochasticProcess:', error);
            const plotElement = document.getElementById('stochastic-plot');
            if (plotElement) {
                plotElement.innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error simulating stochastic process. Please check the console for details.</div>';
            }
        }
    }
    
    static plotStochasticPaths(paths, processType) {
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded');
            document.getElementById('stochastic-plot').innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error: Plotly library not loaded. Please refresh the page.</div>';
            return;
        }
        
        const traces = paths.map((path, index) => ({
            x: path.map((_, i) => i),
            y: path,
            type: 'scatter',
            mode: 'lines',
            name: `Path ${index + 1}`,
            line: { width: 1.5, color: `hsl(${index * 360 / paths.length}, 70%, 50%)` }
        }));
        
        const layout = {
            title: `${processType.replace('-', ' ').toUpperCase()} Simulation`,
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Value' },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' },
            showlegend: paths.length <= 10
        };
        
        try {
            Plotly.newPlot('stochastic-plot', traces, layout, { responsive: true });
        } catch (error) {
            console.error('Error plotting stochastic paths:', error);
            document.getElementById('stochastic-plot').innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error creating plot. Please try again.</div>';
        }
    }
    
    static generateBinomialTree() {
        const S0 = parseFloat(document.getElementById('initial-price').value);
        const u = parseFloat(document.getElementById('up-factor').value);
        const d = parseFloat(document.getElementById('down-factor').value);
        const n = parseInt(document.getElementById('tree-steps').value);
        
        const tree = this.buildBinomialTree(S0, u, d, n);
        this.plotBinomialTree(tree, n);
    }
    
    static buildBinomialTree(S0, u, d, n) {
        const tree = [];
        
        for (let i = 0; i <= n; i++) {
            tree[i] = [];
            for (let j = 0; j <= i; j++) {
                tree[i][j] = S0 * Math.pow(u, i - j) * Math.pow(d, j);
            }
        }
        
        return tree;
    }
    
    static plotBinomialTree(tree, n) {
        const traces = [];
        const annotations = [];
        
        // Plot nodes
        for (let i = 0; i <= n; i++) {
            for (let j = 0; j <= i; j++) {
                const x = i;
                const y = i - 2 * j;
                
                traces.push({
                    x: [x],
                    y: [y],
                    mode: 'markers',
                    marker: { size: 12, color: '#e94560' },
                    showlegend: false
                });
                
                annotations.push({
                    x: x,
                    y: y,
                    text: tree[i][j].toFixed(2),
                    showarrow: false,
                    font: { color: '#ffffff', size: 10 }
                });
                
                // Draw connections
                if (i < n) {
                    // Up move
                    traces.push({
                        x: [x, x + 1],
                        y: [y, y + 1],
                        mode: 'lines',
                        line: { color: '#0f3460', width: 2 },
                        showlegend: false
                    });
                    
                    // Down move
                    traces.push({
                        x: [x, x + 1],
                        y: [y, y - 1],
                        mode: 'lines',
                        line: { color: '#0f3460', width: 2 },
                        showlegend: false
                    });
                }
            }
        }
        
        const layout = {
            title: 'Binomial Price Tree',
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Price Level' },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' },
            annotations: annotations,
            showlegend: false
        };
        
        Plotly.newPlot('binomial-tree', traces, layout, { responsive: true });
    }
    
    static priceOption() {
        const S = parseFloat(document.getElementById('option-spot').value);
        const K = parseFloat(document.getElementById('option-strike').value);
        const T = parseFloat(document.getElementById('option-time').value);
        const r = parseFloat(document.getElementById('option-rate').value) / 100;
        const sigma = parseFloat(document.getElementById('option-vol').value) / 100;
        
        const blackScholesPrice = MathUtils.blackScholes(S, K, T, r, sigma, 'call');
        const binomialPrice = MathUtils.binomialOptionPrice(S, K, T, r, sigma, 100, 'call');
        const greeks = MathUtils.calculateGreeks(S, K, T, r, sigma);
        
        document.getElementById('option-price-result').innerHTML = `
            <h4>Option Pricing Results</h4>
            <p><strong>Binomial Model:</strong> $${binomialPrice.toFixed(4)}</p>
            <p><strong>Convergence to Black-Scholes:</strong> ${Math.abs(blackScholesPrice - binomialPrice) < 0.01 ? '✅' : '❌'}</p>
            <div class="mt-2">
                <h5>Greeks:</h5>
                <p>Delta: ${greeks.delta.toFixed(4)}</p>
                <p>Gamma: ${greeks.gamma.toFixed(4)}</p>
                <p>Theta: ${greeks.theta.toFixed(4)}</p>
                <p>Vega: ${greeks.vega.toFixed(4)}</p>
            </div>
        `;
        
        document.getElementById('black-scholes-comparison').innerHTML = `
            <h4>Black-Scholes Comparison</h4>
            <p><strong>Analytical Price:</strong> $${blackScholesPrice.toFixed(4)}</p>
            <p><strong>Difference:</strong> $${Math.abs(blackScholesPrice - binomialPrice).toFixed(6)}</p>
            <p class="text-muted">As n → ∞, binomial converges to Black-Scholes</p>
        `;
    }
    
    static simulateHedging() {
        const S0 = parseFloat(document.getElementById('hedge-spot').value);
        const optionsPosition = parseInt(document.getElementById('hedge-options').value);
        const frequency = document.getElementById('hedge-frequency').value;
        
        // Simulate stock price path
        const T = 0.25; // 3 months
        const sigma = 0.2;
        const r = 0.05;
        const K = S0; // ATM option
        
        let rehedgeSteps;
        switch (frequency) {
            case 'daily': rehedgeSteps = 63; break; // ~3 months of trading days
            case 'hourly': rehedgeSteps = 63 * 8; break; // 8 hours per day
            case 'continuous': rehedgeSteps = 1000; break;
        }
        
        const dt = T / rehedgeSteps;
        const stockPath = [S0];
        const hedgeRatio = [];
        const portfolioValue = [];
        
        let currentStock = S0;
        
        for (let i = 0; i <= rehedgeSteps; i++) {
            const timeToExpiry = T - i * dt;
            
            if (timeToExpiry > 0) {
                const greeks = MathUtils.calculateGreeks(currentStock, K, timeToExpiry, r, sigma);
                hedgeRatio.push(greeks.delta * optionsPosition);
                
                const optionValue = MathUtils.blackScholes(currentStock, K, timeToExpiry, r, sigma, 'call');
                const portfolioVal = optionsPosition * optionValue - hedgeRatio[i] * currentStock;
                portfolioValue.push(portfolioVal);
            }
            
            if (i < rehedgeSteps) {
                const dW = MathUtils.randomNormal(0, Math.sqrt(dt));
                currentStock = currentStock * Math.exp((r - 0.5 * sigma * sigma) * dt + sigma * dW);
                stockPath.push(currentStock);
            }
        }
        
        this.plotHedgingSimulation(stockPath, hedgeRatio, portfolioValue, frequency);
    }
    
    static plotHedgingSimulation(stockPath, hedgeRatio, portfolioValue, frequency) {
        const timeAxis = stockPath.map((_, i) => i);
        
        const traces = [
            {
                x: timeAxis,
                y: stockPath,
                type: 'scatter',
                mode: 'lines',
                name: 'Stock Price',
                yaxis: 'y',
                line: { color: '#e94560' }
            },
            {
                x: timeAxis.slice(0, -1),
                y: hedgeRatio,
                type: 'scatter',
                mode: 'lines',
                name: 'Hedge Ratio (Delta)',
                yaxis: 'y2',
                line: { color: '#0f3460' }
            }
        ];
        
        const layout = {
            title: `Delta Hedging Simulation (${frequency} rebalancing)`,
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Stock Price ($)', side: 'left' },
            yaxis2: {
                title: 'Hedge Ratio',
                side: 'right',
                overlaying: 'y'
            },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' }
        };
        
        Plotly.newPlot('hedging-plot', traces, layout, { responsive: true });
    }
    
    static runMonteCarlo() {
        try {
            const numSimsElement = document.getElementById('mc-simulations');
            const optionTypeElement = document.getElementById('mc-option-type');
            
            const numSims = numSimsElement ? parseInt(numSimsElement.value) || 10000 : 10000;
            const optionType = optionTypeElement ? optionTypeElement.value : 'european-call';
            
            console.log(`Running Monte Carlo with ${numSims} simulations for ${optionType}`);
            
            // Standard parameters
            const S = 100, K = 100, T = 0.25, r = 0.05, sigma = 0.2;
            
            let result;
            
            switch (optionType) {
                case 'european-call':
                    result = MonteCarloEngine.europeanOption(S, K, T, r, sigma, numSims, 'call');
                    break;
                case 'european-put':
                    result = MonteCarloEngine.europeanOption(S, K, T, r, sigma, numSims, 'put');
                    break;
                case 'asian-call':
                    result = { price: MonteCarloEngine.asianOption(S, K, T, r, sigma, numSims, 50, 'call') };
                    break;
                case 'barrier-call':
                    result = { price: MonteCarloEngine.barrierOption(S, K, T, r, sigma, 110, numSims, 'up-and-out', 'call') };
                    break;
                default:
                    result = MonteCarloEngine.europeanOption(S, K, T, r, sigma, numSims, 'call');
            }
            
            const analytical = MathUtils.blackScholes(S, K, T, r, sigma, optionType.includes('call') ? 'call' : 'put');
            
            const resultElement = document.getElementById('monte-carlo-result');
            if (resultElement) {
                resultElement.innerHTML = `
                    <h4>Monte Carlo Results</h4>
                    <p><strong>Simulated Price:</strong> $${result.price.toFixed(4)}</p>
                    ${result.standardError ? `<p><strong>Standard Error:</strong> $${result.standardError.toFixed(4)}</p>` : ''}
                    ${optionType.includes('european') ? `<p><strong>Analytical Price:</strong> $${analytical.toFixed(4)}</p>` : ''}
                    <p><strong>Number of Simulations:</strong> ${numSims.toLocaleString()}</p>
                `;
            }
            
            if (result.payoffs) {
                this.plotMonteCarloResults(result.payoffs);
            }
        } catch (error) {
            console.error('Error in runMonteCarlo:', error);
            const resultElement = document.getElementById('monte-carlo-result');
            if (resultElement) {
                resultElement.innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error running Monte Carlo simulation. Please check the console for details.</div>';
            }
        }
    }
    
    static plotMonteCarloResults(payoffs) {
        const trace = {
            x: payoffs,
            type: 'histogram',
            nbinsx: 50,
            marker: { color: '#0f3460' },
            name: 'Payoff Distribution'
        };
        
        const layout = {
            title: 'Monte Carlo Payoff Distribution',
            xaxis: { title: 'Payoff ($)' },
            yaxis: { title: 'Frequency' },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' }
        };
        
        Plotly.newPlot('monte-carlo-plot', [trace], layout, { responsive: true });
    }
    
    static simulateOrderBook() {
        const orderSize = parseInt(document.getElementById('order-size').value);
        const volatility = parseFloat(document.getElementById('market-volatility').value);
        
        // Simulate order book with bid-ask spread
        const midPrice = 100;
        const baseSpread = 0.02 * volatility; // Spread increases with volatility
        
        const bids = [];
        const asks = [];
        
        // Generate order book levels
        for (let i = 1; i <= 10; i++) {
            const bidPrice = midPrice - baseSpread / 2 - (i - 1) * 0.01;
            const askPrice = midPrice + baseSpread / 2 + (i - 1) * 0.01;
            
            const bidSize = Math.floor(Math.random() * 1000 + 500);
            const askSize = Math.floor(Math.random() * 1000 + 500);
            
            bids.push({ price: bidPrice, size: bidSize });
            asks.push({ price: askPrice, size: askSize });
        }
        
        // Calculate market impact
        let remainingSize = orderSize;
        let totalCost = 0;
        let impactedLevels = [];
        
        for (let i = 0; i < asks.length && remainingSize > 0; i++) {
            const levelSize = Math.min(remainingSize, asks[i].size);
            totalCost += levelSize * asks[i].price;
            remainingSize -= levelSize;
            
            impactedLevels.push({
                price: asks[i].price,
                size: levelSize,
                remaining: asks[i].size - levelSize
            });
        }
        
        const avgPrice = totalCost / (orderSize - remainingSize);
        const slippage = avgPrice - midPrice;
        
        this.plotOrderBook(bids, asks, impactedLevels, midPrice, avgPrice, slippage);
    }
    
    static plotOrderBook(bids, asks, impactedLevels, midPrice, avgPrice, slippage) {
        const bidTrace = {
            x: bids.map(b => b.size),
            y: bids.map(b => b.price),
            type: 'bar',
            orientation: 'h',
            name: 'Bids',
            marker: { color: '#4caf50' },
            text: bids.map(b => `$${b.price.toFixed(2)}`),
            textposition: 'auto'
        };
        
        const askTrace = {
            x: asks.map(a => a.size),
            y: asks.map(a => a.price),
            type: 'bar',
            orientation: 'h',
            name: 'Asks',
            marker: { color: '#f44336' },
            text: asks.map(a => `$${a.price.toFixed(2)}`),
            textposition: 'auto'
        };
        
        const layout = {
            title: `Order Book Simulation<br>Slippage: $${slippage.toFixed(4)} | Avg Price: $${avgPrice.toFixed(4)}`,
            xaxis: { title: 'Order Size' },
            yaxis: { title: 'Price ($)' },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' },
            barmode: 'relative'
        };
        
        Plotly.newPlot('orderbook-plot', [bidTrace, askTrace], layout, { responsive: true });
    }
    
    static showCaseStudy(caseType) {
        const caseStudyDetail = document.getElementById('case-study-detail');
        const caseStudyContent = document.getElementById('case-study-content');
        
        const cases = {
            ltcm: {
                title: 'Long-Term Capital Management (1998)',
                content: `
                    <h4>The Rise and Fall of LTCM</h4>
                    <p><strong>Background:</strong> Founded in 1994 by John Meriwether with Nobel laureates Myron Scholes and Robert Merton on the board.</p>
                    
                    <h5>The Strategy</h5>
                    <ul>
                        <li>Convergence trading: betting on price differences between similar securities</li>
                        <li>Fixed-income arbitrage: exploiting yield curve anomalies</li>
                        <li>Equity volatility trading: statistical arbitrage strategies</li>
                    </ul>
                    
                    <h5>The Mathematics</h5>
                    <p>LTCM used sophisticated models based on:</p>
                    <ul>
                        <li>Mean reversion assumptions</li>
                        <li>Historical correlation patterns</li>
                        <li>Value-at-Risk (VaR) models with normal distributions</li>
                    </ul>
                    
                    <h5>The Collapse</h5>
                    <p><strong>Leverage:</strong> 25:1 ratio amplified losses</p>
                    <p><strong>Liquidity Crisis:</strong> Russian default caused flight to quality</p>
                    <p><strong>Model Failure:</strong> Correlations approached 1 during crisis</p>
                    
                    <h5>Lessons Learned</h5>
                    <ul>
                        <li>Tail risk and fat-tailed distributions</li>
                        <li>Liquidity risk in crisis periods</li>
                        <li>Importance of stress testing</li>
                        <li>Limits of diversification during systemic events</li>
                    </ul>
                `
            },
            cip: {
                title: 'Covered Interest Parity Breakdown',
                content: `
                    <h4>The Theory vs. Reality</h4>
                    <p><strong>Covered Interest Parity (CIP):</strong> Interest rate differentials should equal forward premiums</p>
                    
                    <h5>The Formula</h5>
                    <p>F/S = (1 + r_domestic)/(1 + r_foreign)</p>
                    
                    <h5>Pre-2008: Theory Held</h5>
                    <ul>
                        <li>Deviations were small and temporary</li>
                        <li>Arbitrage quickly eliminated discrepancies</li>
                        <li>Efficient markets assumption validated</li>
                    </ul>
                    
                    <h5>Post-2008: Persistent Deviations</h5>
                    <ul>
                        <li>Basel III regulations increased bank funding costs</li>
                        <li>Balance sheet constraints limited arbitrage</li>
                        <li>Dollar funding premiums emerged</li>
                    </ul>
                    
                    <h5>New Opportunities</h5>
                    <ul>
                        <li>Cross-currency basis swaps</li>
                        <li>FX forward arbitrage</li>
                        <li>Regulatory arbitrage strategies</li>
                    </ul>
                `
            },
            flash: {
                title: 'Flash Crash of May 6, 2010',
                content: `
                    <h4>The Event</h4>
                    <p><strong>Timeline:</strong> 2:45 PM - Dow Jones drops 1000 points in minutes</p>
                    <p><strong>Recovery:</strong> Most losses recovered within 20 minutes</p>
                    
                    <h5>The Trigger</h5>
                    <ul>
                        <li>Large sell order of E-mini S&P 500 futures</li>
                        <li>Algorithm executed without regard to price or time</li>
                        <li>Order size: $4.1 billion worth of contracts</li>
                    </ul>
                    
                    <h5>The Cascade</h5>
                    <ul>
                        <li>High-frequency traders withdrew liquidity</li>
                        <li>Cross-market arbitrage broke down</li>
                        <li>ETFs traded at massive discounts</li>
                    </ul>
                    
                    <h5>Systemic Issues Revealed</h5>
                    <ul>
                        <li>Market fragmentation across venues</li>
                        <li>Lack of coordination between markets</li>
                        <li>Algorithmic trading amplification</li>
                    </ul>
                    
                    <h5>Regulatory Response</h5>
                    <ul>
                        <li>Circuit breakers implemented</li>
                        <li>Limit up/limit down rules</li>
                        <li>Enhanced market surveillance</li>
                    </ul>
                `
            }
        };
        
        const selectedCase = cases[caseType];
        if (selectedCase) {
            caseStudyContent.innerHTML = selectedCase.content;
            caseStudyDetail.classList.remove('hidden');
            
            // Simulate some data for the case study
            this.simulateCaseStudyData(caseType);
        }
    }
    
    static simulateCaseStudyData(caseType) {
        // Generate relevant simulation data for each case study
        const container = document.getElementById('case-study-simulation');
        
        switch (caseType) {
            case 'ltcm':
                this.simulateLTCMStrategy();
                break;
            case 'cip':
                this.simulateCIPDeviations();
                break;
            case 'flash':
                this.simulateFlashCrash();
                break;
        }
    }
    
    static simulateLTCMStrategy() {
        // Simulate convergence trading strategy
        const days = 1000;
        const spread = [];
        const pnl = [];
        
        let currentSpread = 0.5; // 50 bps initial spread
        let cumulativePnL = 0;
        
        for (let i = 0; i < days; i++) {
            // Mean-reverting spread with occasional jumps
            const meanReversion = -0.1 * currentSpread;
            const randomShock = MathUtils.randomNormal(0, 0.05);
            const jumpRisk = Math.random() < 0.001 ? MathUtils.randomNormal(0, 2) : 0; // 0.1% chance of large jump
            
            currentSpread += meanReversion + randomShock + jumpRisk;
            spread.push(currentSpread);
            
            // P&L from convergence trade (short the spread)
            const dailyPnL = -currentSpread * 0.01; // Simplified P&L
            cumulativePnL += dailyPnL;
            pnl.push(cumulativePnL);
        }
        
        const traces = [
            {
                x: Array.from({length: days}, (_, i) => i),
                y: spread,
                type: 'scatter',
                mode: 'lines',
                name: 'Spread (bps)',
                yaxis: 'y',
                line: { color: '#e94560' }
            },
            {
                x: Array.from({length: days}, (_, i) => i),
                y: pnl,
                type: 'scatter',
                mode: 'lines',
                name: 'Cumulative P&L',
                yaxis: 'y2',
                line: { color: '#0f3460' }
            }
        ];
        
        const layout = {
            title: 'LTCM-Style Convergence Trading Simulation',
            xaxis: { title: 'Days' },
            yaxis: { title: 'Spread (bps)', side: 'left' },
            yaxis2: {
                title: 'P&L',
                side: 'right',
                overlaying: 'y'
            },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' }
        };
        
        Plotly.newPlot('case-study-simulation', traces, layout, { responsive: true });
    }
    
    static simulateCIPDeviations() {
        // Simulate CIP deviations over time
        const months = 180; // 15 years
        const deviations = [];
        const crisis = 60; // Crisis starts at month 60
        
        for (let i = 0; i < months; i++) {
            let deviation;
            if (i < crisis) {
                // Pre-crisis: small deviations
                deviation = MathUtils.randomNormal(0, 2); // 2 bps std
            } else {
                // Post-crisis: larger, persistent deviations
                const trend = 5; // 5 bps average deviation
                deviation = trend + MathUtils.randomNormal(0, 8); // 8 bps std
            }
            deviations.push(deviation);
        }
        
        const trace = {
            x: Array.from({length: months}, (_, i) => i),
            y: deviations,
            type: 'scatter',
            mode: 'lines',
            name: 'CIP Deviation (bps)',
            line: { color: '#e94560' }
        };
        
        const layout = {
            title: 'Covered Interest Parity Deviations (2008 Crisis Impact)',
            xaxis: { title: 'Months from Start' },
            yaxis: { title: 'Deviation (basis points)' },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' },
            shapes: [{
                type: 'line',
                x0: crisis,
                x1: crisis,
                y0: Math.min(...deviations),
                y1: Math.max(...deviations),
                line: { color: '#ff9800', width: 2, dash: 'dash' }
            }],
            annotations: [{
                x: crisis,
                y: Math.max(...deviations) * 0.8,
                text: '2008 Crisis',
                showarrow: true,
                arrowcolor: '#ff9800'
            }]
        };
        
        Plotly.newPlot('case-study-simulation', [trace], layout, { responsive: true });
    }
    
    static simulateFlashCrash() {
        // Simulate flash crash price action
        const minutes = 60; // 1 hour around the crash
        const crashStart = 30; // Crash starts at minute 30
        const prices = [];
        const volume = [];
        
        let currentPrice = 1150; // S&P 500 level
        
        for (let i = 0; i < minutes; i++) {
            if (i < crashStart) {
                // Normal trading
                currentPrice += MathUtils.randomNormal(0, 2);
                volume.push(Math.random() * 1000 + 500);
            } else if (i < crashStart + 10) {
                // Flash crash period
                currentPrice += MathUtils.randomNormal(-20, 10); // Sharp decline
                volume.push(Math.random() * 5000 + 2000); // High volume
            } else {
                // Recovery
                const recovery = (1150 - currentPrice) * 0.3; // 30% recovery each minute
                currentPrice += recovery + MathUtils.randomNormal(0, 5);
                volume.push(Math.random() * 2000 + 1000);
            }
            
            prices.push(currentPrice);
        }
        
        const traces = [
            {
                x: Array.from({length: minutes}, (_, i) => i),
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: 'S&P 500 Price',
                yaxis: 'y',
                line: { color: '#e94560', width: 3 }
            },
            {
                x: Array.from({length: minutes}, (_, i) => i),
                y: volume,
                type: 'bar',
                name: 'Volume',
                yaxis: 'y2',
                marker: { color: '#0f3460', opacity: 0.6 }
            }
        ];
        
        const layout = {
            title: 'Flash Crash Simulation - May 6, 2010',
            xaxis: { title: 'Minutes' },
            yaxis: { title: 'S&P 500 Price', side: 'left' },
            yaxis2: {
                title: 'Volume',
                side: 'right',
                overlaying: 'y'
            },
            paper_bgcolor: '#1e1e2e',
            plot_bgcolor: '#2a2a3e',
            font: { color: '#ffffff' },
            shapes: [{
                type: 'rect',
                x0: crashStart,
                x1: crashStart + 10,
                y0: Math.min(...prices),
                y1: Math.max(...prices),
                fillcolor: 'rgba(244, 67, 54, 0.2)',
                line: { width: 0 }
            }],
            annotations: [{
                x: crashStart + 5,
                y: Math.min(...prices) + 50,
                text: 'Flash Crash Period',
                showarrow: false,
                font: { color: '#ffffff', size: 12 }
            }]
        };
        
        Plotly.newPlot('case-study-simulation', traces, layout, { responsive: true });
    }
    
    static generateNewProblem() {
        try {
            const difficultyElement = document.getElementById('problem-difficulty');
            const topicElement = document.getElementById('problem-topic');
            
            const difficulty = difficultyElement ? difficultyElement.value : 'easy';
            const topic = topicElement ? topicElement.value : 'arbitrage';
            
            console.log(`Generating ${difficulty} problem on ${topic}`);
            
            currentProblem = ProblemGenerator.generateProblem(difficulty, topic);
            
            const problemStatementElement = document.getElementById('problem-statement');
            if (problemStatementElement) {
                problemStatementElement.innerHTML = `
                    <h4>Problem ${problemHistory.length + 1} (${currentProblem.difficulty} - ${currentProblem.topic})</h4>
                    <p>${currentProblem.question}</p>
                `;
            }
            
            const answerInputElement = document.getElementById('answer-input');
            if (answerInputElement) {
                answerInputElement.value = '';
                answerInputElement.focus();
            }
            
            const feedbackElement = document.getElementById('problem-feedback');
            if (feedbackElement) {
                feedbackElement.innerHTML = '';
            }
        } catch (error) {
            console.error('Error generating new problem:', error);
            const problemStatementElement = document.getElementById('problem-statement');
            if (problemStatementElement) {
                problemStatementElement.innerHTML = '<div style="color: #ff6b6b; text-align: center; padding: 20px;">Error generating problem. Please try again.</div>';
            }
        }
    }
    
    static checkAnswer() {
        if (!currentProblem) {
            alert('Please generate a problem first!');
            return;
        }
        
        const userAnswer = parseFloat(document.getElementById('answer-input').value);
        const tolerance = Math.abs(currentProblem.answer) * 0.01; // 1% tolerance
        const isCorrect = Math.abs(userAnswer - currentProblem.answer) <= tolerance;
        
        const feedback = document.getElementById('problem-feedback');
        
        if (isCorrect) {
            feedback.innerHTML = `
                <div class="text-success">
                    <h5>✅ Correct!</h5>
                    <p><strong>Your answer:</strong> ${userAnswer.toFixed(4)}</p>
                    <p><strong>Expected:</strong> ${currentProblem.answer.toFixed(4)}</p>
                    <p><strong>Explanation:</strong> ${currentProblem.explanation}</p>
                </div>
            `;
        } else {
            feedback.innerHTML = `
                <div class="text-danger">
                    <h5>❌ Incorrect</h5>
                    <p><strong>Your answer:</strong> ${userAnswer.toFixed(4)}</p>
                    <p><strong>Correct answer:</strong> ${currentProblem.answer.toFixed(4)}</p>
                    <p><strong>Explanation:</strong> ${currentProblem.explanation}</p>
                </div>
            `;
        }
        
        // Add to history
        problemHistory.push({
            ...currentProblem,
            userAnswer: userAnswer,
            correct: isCorrect,
            timestamp: new Date()
        });
        
        this.updateProblemHistory();
    }
    
    static updateProblemHistory() {
        const historyContainer = document.getElementById('problem-history-list');
        
        if (!historyContainer) {
            console.error('Problem history container not found');
            return;
        }
        
        if (problemHistory.length === 0) {
            historyContainer.innerHTML = '<p class="text-muted">No problems solved yet.</p>';
            return;
        }
        
        const correctCount = problemHistory.filter(p => p.correct).length;
        const accuracy = (correctCount / problemHistory.length * 100).toFixed(1);
        
        historyContainer.innerHTML = `
            <h5>Performance Summary</h5>
            <p><strong>Problems Solved:</strong> ${problemHistory.length}</p>
            <p><strong>Accuracy:</strong> ${accuracy}% (${correctCount}/${problemHistory.length})</p>
            
            <h6>Recent Problems:</h6>
            <div class="problem-history-list">
                ${problemHistory.slice(-5).reverse().map((problem, index) => `
                    <div class="problem-history-item ${problem.correct ? 'correct' : 'incorrect'}">
                        <span class="problem-number">#${problemHistory.length - index}</span>
                        <span class="problem-topic">${problem.topic}</span>
                        <span class="problem-result">${problem.correct ? '✅' : '❌'}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }
}

// Market Data Simulator
class MarketDataSimulator {
    static generateRealisticPrices(symbol, days = 252) {
        const prices = [100]; // Start at $100
        const returns = [];
        
        // Market parameters
        const annualDrift = 0.08; // 8% annual return
        const annualVol = 0.25; // 25% annual volatility
        const dt = 1 / 252; // Daily time step
        
        for (let i = 1; i < days; i++) {
            const drift = annualDrift * dt;
            const diffusion = annualVol * Math.sqrt(dt) * MathUtils.randomNormal();
            const return_ = drift + diffusion;
            
            returns.push(return_);
            prices.push(prices[i-1] * Math.exp(return_));
        }
        
        return { prices, returns };
    }
    
    static generateCorrelatedAssets(numAssets, days = 252, correlation = 0.3) {
        const assets = [];
        const baseReturns = [];
        
        // Generate base random returns
        for (let i = 0; i < days - 1; i++) {
            baseReturns.push(MathUtils.randomNormal());
        }
        
        for (let asset = 0; asset < numAssets; asset++) {
            const prices = [100];
            const returns = [];
            
            for (let i = 0; i < days - 1; i++) {
                // Correlated return = correlation * base + sqrt(1-correlation^2) * independent
                const independentReturn = MathUtils.randomNormal();
                const correlatedReturn = correlation * baseReturns[i] + 
                    Math.sqrt(1 - correlation * correlation) * independentReturn;
                
                const scaledReturn = 0.08 / 252 + 0.25 / Math.sqrt(252) * correlatedReturn;
                returns.push(scaledReturn);
                prices.push(prices[i] * Math.exp(scaledReturn));
            }
            
            assets.push({ prices, returns, symbol: `Asset${asset + 1}` });
        }
        
        return assets;
    }
}

// Risk Management Tools
class RiskMetrics {
    static calculateVaR(returns, confidence = 0.05) {
        const sortedReturns = [...returns].sort((a, b) => a - b);
        const index = Math.floor(confidence * sortedReturns.length);
        return sortedReturns[index];
    }
    
    static calculateCVaR(returns, confidence = 0.05) {
        const sortedReturns = [...returns].sort((a, b) => a - b);
        const cutoff = Math.floor(confidence * sortedReturns.length);
        const tailReturns = sortedReturns.slice(0, cutoff);
        return tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length;
    }
    
    static calculateMaxDrawdown(prices) {
        let maxDrawdown = 0;
        let peak = prices[0];
        
        for (let i = 1; i < prices.length; i++) {
            if (prices[i] > peak) {
                peak = prices[i];
            }
            
            const drawdown = (peak - prices[i]) / peak;
            if (drawdown > maxDrawdown) {
                maxDrawdown = drawdown;
            }
        }
        
        return maxDrawdown;
    }
    
    static calculateSharpeRatio(returns, riskFreeRate = 0.02) {
        const excessReturns = returns.map(r => r - riskFreeRate / 252);
        const meanExcess = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
        const stdExcess = Math.sqrt(
            excessReturns.reduce((sum, r) => sum + Math.pow(r - meanExcess, 2), 0) / (excessReturns.length - 1)
        );
        
        return (meanExcess * 252) / (stdExcess * Math.sqrt(252));
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    UIController.init();
    
    // Add some initial content to empty sections
    const welcomeMessage = document.getElementById('welcome-message');
    if (welcomeMessage) {
        welcomeMessage.innerHTML = `
            <p>Welcome to QuantZ Week 1 Orientation! This interactive platform will guide you through the foundational concepts of quantitative finance.</p>
            <p>Use the navigation above to explore different sections, or scroll down to begin with the core pillars of quant finance.</p>
        `;
    }
    
    // Initialize problem system
    const problemStatement = document.getElementById('problem-statement');
    if (problemStatement) {
        problemStatement.innerHTML = '<p class="text-muted">Click "Generate New Problem" to start practicing!</p>';
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Enter key in answer input
        if (e.target.id === 'answer-input' && e.key === 'Enter') {
            UIController.checkAnswer();
        }
        
        // Ctrl+G for new problem
        if (e.ctrlKey && e.key === 'g') {
            e.preventDefault();
            UIController.generateNewProblem();
        }
    });
    
    console.log('QuantZ Week 1 Orientation Platform Initialized');
    console.log('Available modules: Arbitrage Detection, Risk-Neutral Pricing, Stochastic Processes, Option Pricing, Monte Carlo, Market Microstructure, Case Studies, Practice Problems');
});