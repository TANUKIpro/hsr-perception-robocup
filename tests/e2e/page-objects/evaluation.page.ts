import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Evaluation Page Object
 *
 * Page: 6_Evaluation.py
 * Features:
 * - Model verification interface
 * - mAP calculation display
 * - Inference timing measurements
 * - Visual prediction verification
 * - Robustness testing
 */
export class EvaluationPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Evaluation';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Evaluation page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Main Tabs
  // ============================================

  /**
   * Click Run Evaluation tab
   */
  async clickRunEvaluationTab(): Promise<void> {
    await this.clickTab('Run Evaluation');
  }

  /**
   * Click Results tab
   */
  async clickResultsTab(): Promise<void> {
    await this.clickTab('Results');
  }

  /**
   * Click Visual Test tab
   */
  async clickVisualTestTab(): Promise<void> {
    await this.clickTab('Visual Test');
  }

  /**
   * Click Robustness Test tab
   */
  async clickRobustnessTestTab(): Promise<void> {
    await this.clickTab('Robustness Test');
  }

  /**
   * Click Xtion Live Test tab
   */
  async clickXtionLiveTestTab(): Promise<void> {
    await this.clickTab('Xtion Live Test');
  }

  // ============================================
  // Run Evaluation Tab
  // ============================================

  /**
   * Select model for evaluation
   */
  async selectModel(modelName: string): Promise<void> {
    await this.clickRunEvaluationTab();
    await this.selectOption('Model', modelName);
  }

  /**
   * Select dataset for evaluation
   */
  async selectDataset(datasetName: string): Promise<void> {
    await this.clickRunEvaluationTab();
    await this.selectOption('Dataset', datasetName);
  }

  /**
   * Start evaluation
   */
  async startEvaluation(): Promise<void> {
    await this.clickButton('Run Evaluation');
  }

  /**
   * Check if evaluation is in progress
   */
  async isEvaluationInProgress(): Promise<boolean> {
    const progressBar = this.selectors.progressBar;
    return await progressBar.isVisible().catch(() => false);
  }

  /**
   * Wait for evaluation to complete
   */
  async waitForEvaluationComplete(timeout: number = 120000): Promise<void> {
    await this.wait.waitForProgressComplete({ timeout });
  }

  // ============================================
  // Competition Requirements
  // ============================================

  /**
   * Get target mAP@50 requirement
   */
  async getTargetMap(): Promise<string> {
    return await this.getMetricValue('Target mAP@50');
  }

  /**
   * Get target inference time requirement
   */
  async getTargetInference(): Promise<string> {
    return await this.getMetricValue('Target Inference');
  }

  // ============================================
  // Results Tab
  // ============================================

  /**
   * Get evaluation results
   */
  async getEvaluationResults(): Promise<{
    map50?: string;
    inferenceTime?: string;
    meetsRequirements?: boolean;
  }> {
    await this.clickResultsTab();

    const results: {
      map50?: string;
      inferenceTime?: string;
      meetsRequirements?: boolean;
    } = {};

    // Try to get mAP value
    try {
      results.map50 = await this.getMetricValue('mAP@50');
    } catch {
      // Metric not available
    }

    // Try to get inference time
    try {
      results.inferenceTime = await this.getMetricValue('Inference Time');
    } catch {
      // Metric not available
    }

    // Check for success/fail indicators
    const successIndicator = this.page.getByText(/Meets Requirements|Pass/i);
    results.meetsRequirements = await successIndicator.isVisible().catch(() => false);

    return results;
  }

  // ============================================
  // Visual Test Tab
  // ============================================

  /**
   * Select image source for visual test
   */
  async selectImageSource(source: 'Image' | 'Camera' | 'Video'): Promise<void> {
    await this.clickVisualTestTab();
    await this.clickTab(source);
  }

  /**
   * Upload test image
   */
  async uploadTestImage(imagePath: string): Promise<void> {
    await this.clickVisualTestTab();
    await this.selectImageSource('Image');
    await this.uploadFile(imagePath);
  }

  /**
   * Run visual prediction
   */
  async runPrediction(): Promise<void> {
    await this.clickButton('Run Prediction');
  }

  /**
   * Check if prediction results are displayed
   */
  async arePredictionResultsVisible(): Promise<boolean> {
    const resultsImage = this.selectors.image;
    return await resultsImage.isVisible().catch(() => false);
  }

  // ============================================
  // Robustness Test Tab
  // ============================================

  /**
   * Get robustness test configuration section
   */
  async getRobustnessConfigSection(): Promise<Locator> {
    await this.clickRobustnessTestTab();
    return this.page.getByText('Robustness Test').locator('..');
  }

  /**
   * Configure brightness test range
   */
  async setBrightnessRange(min: number, max: number): Promise<void> {
    await this.clickRobustnessTestTab();
    // Slider configuration would go here
  }

  /**
   * Run robustness test
   */
  async runRobustnessTest(): Promise<void> {
    await this.clickRobustnessTestTab();
    await this.clickButton('Run Robustness Test');
  }

  /**
   * Wait for robustness test to complete
   */
  async waitForRobustnessTestComplete(timeout: number = 180000): Promise<void> {
    await this.wait.waitForProgressComplete({ timeout });
  }

  // ============================================
  // Xtion Live Test Tab
  // ============================================

  /**
   * Check if Xtion camera is available
   */
  async isXtionAvailable(): Promise<boolean> {
    await this.clickXtionLiveTestTab();
    const unavailableMsg = this.page.getByText(/not available|not connected/i);
    return !(await unavailableMsg.isVisible().catch(() => false));
  }

  /**
   * Start live test
   */
  async startLiveTest(): Promise<void> {
    await this.clickXtionLiveTestTab();
    await this.clickButton('Start');
  }

  /**
   * Stop live test
   */
  async stopLiveTest(): Promise<void> {
    await this.clickButton('Stop');
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert evaluation tabs are visible
   */
  async expectEvaluationTabsVisible(): Promise<void> {
    await expect(this.page.getByRole('tab', { name: /Run Evaluation/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Results/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Visual Test/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Robustness Test/i })).toBeVisible();
  }

  /**
   * Assert model selection is visible
   */
  async expectModelSelectionVisible(): Promise<void> {
    await this.clickRunEvaluationTab();
    await expect(this.page.getByText('Model')).toBeVisible();
  }

  /**
   * Assert dataset selection is visible
   */
  async expectDatasetSelectionVisible(): Promise<void> {
    await this.clickRunEvaluationTab();
    await expect(this.page.getByText('Dataset')).toBeVisible();
  }

  /**
   * Assert competition requirements are displayed
   */
  async expectCompetitionRequirementsVisible(): Promise<void> {
    await expect(this.selectors.metric('Target mAP@50')).toBeVisible();
    await expect(this.selectors.metric('Target Inference')).toBeVisible();
  }

  /**
   * Assert evaluation success
   */
  async expectEvaluationSuccess(): Promise<void> {
    await this.expectSuccessAlert(/completed|finished/i);
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await this.expectEvaluationTabsVisible();
  }
}
