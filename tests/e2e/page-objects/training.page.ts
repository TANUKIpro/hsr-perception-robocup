import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Training Page Object
 *
 * Page: 5_Training.py
 * Features:
 * - Training configuration (dataset, model, epochs, etc.)
 * - Real-time training charts
 * - TensorBoard integration
 * - GPU status display
 * - Model management
 */
export class TrainingPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Training';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Training page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Main Tabs
  // ============================================

  /**
   * Click Start Training tab
   */
  async clickStartTrainingTab(): Promise<void> {
    await this.clickTabExact('⊞ Start Training');
  }

  /**
   * Click Models tab
   */
  async clickModelsTab(): Promise<void> {
    await this.clickTabExact('◎ Models');
  }

  /**
   * Click History tab
   */
  async clickHistoryTab(): Promise<void> {
    await this.clickTabExact('📜 History');
  }

  // ============================================
  // Configuration Sub-tabs
  // ============================================

  /**
   * Click Dataset sub-tab
   */
  async clickDatasetSubTab(): Promise<void> {
    await this.clickTabExact('📂 Dataset');
  }

  /**
   * Click Synthetic sub-tab
   */
  async clickSyntheticSubTab(): Promise<void> {
    await this.clickTabExact('🎨 Synthetic');
  }

  /**
   * Click Model sub-tab
   */
  async clickModelSubTab(): Promise<void> {
    await this.clickTabExact('🤖 Model');
  }

  /**
   * Click Advanced sub-tab
   */
  async clickAdvancedSubTab(): Promise<void> {
    await this.clickTabExact('⚙️ Advanced');
  }

  /**
   * Click Others sub-tab
   */
  async clickOthersSubTab(): Promise<void> {
    await this.clickTabExact('📦 Others');
  }

  // ============================================
  // Dataset Configuration
  // ============================================

  /**
   * Select dataset
   */
  async selectDataset(datasetName: string): Promise<void> {
    await this.clickDatasetSubTab();
    await this.selectOption('Dataset', datasetName);
  }

  /**
   * Get available datasets
   */
  async getAvailableDatasets(): Promise<string[]> {
    await this.clickDatasetSubTab();
    const selectbox = this.selectors.selectbox('Dataset');
    await selectbox.click();

    const options = this.page.locator('[data-baseweb="menu"] li');
    const count = await options.count();
    const datasets: string[] = [];

    for (let i = 0; i < count; i++) {
      const text = await options.nth(i).textContent();
      if (text) datasets.push(text.trim());
    }

    // Close dropdown
    await this.page.keyboard.press('Escape');
    return datasets;
  }

  // ============================================
  // Model Configuration
  // ============================================

  /**
   * Select base model
   */
  async selectBaseModel(modelName: string): Promise<void> {
    await this.clickModelSubTab();
    await this.selectOption('Base Model', modelName);
  }

  /**
   * Set number of epochs
   */
  async setEpochs(epochs: number): Promise<void> {
    await this.clickModelSubTab();
    const slider = this.selectors.slider('Epochs');
    // Slider interaction - click and use keyboard
    await slider.click();
    // Clear and type the value
    await this.page.keyboard.press('Control+a');
    await this.page.keyboard.type(String(epochs));
    await this.wait.waitForRerun();
  }

  /**
   * Set batch size
   */
  async setBatchSize(batchSize: number): Promise<void> {
    await this.clickModelSubTab();
    await this.fillNumberInput('Batch Size', batchSize);
  }

  /**
   * Set image size
   */
  async setImageSize(size: number): Promise<void> {
    await this.clickModelSubTab();
    await this.fillNumberInput('Image Size', size);
  }

  // ============================================
  // Training Control
  // ============================================

  /**
   * Start training
   */
  async startTraining(): Promise<void> {
    await this.clickButton('Start Training');
  }

  /**
   * Cancel training
   */
  async cancelTraining(): Promise<void> {
    await this.clickButton('Cancel');
  }

  /**
   * Check if training is in progress
   */
  async isTrainingInProgress(): Promise<boolean> {
    const cancelButton = this.page.getByRole('button', { name: 'Cancel' });
    return await cancelButton.isVisible().catch(() => false);
  }

  /**
   * Check if start button is enabled
   */
  async isStartButtonEnabled(): Promise<boolean> {
    const button = this.page.getByRole('button', { name: 'Start Training' });
    if (await button.isVisible().catch(() => false)) {
      return await button.isEnabled();
    }
    return false;
  }

  // ============================================
  // GPU & Hardware
  // ============================================

  /**
   * Get GPU status information
   */
  async getGpuStatus(): Promise<{ visible: boolean; info?: string }> {
    await this.clickOthersSubTab();
    const gpuSection = this.page.getByText('Hardware & GPU');

    if (await gpuSection.isVisible().catch(() => false)) {
      const info = await gpuSection.locator('..').textContent();
      return { visible: true, info: info || undefined };
    }
    return { visible: false };
  }

  /**
   * Check if CUDA is available
   */
  async isCudaAvailable(): Promise<boolean> {
    const { info } = await this.getGpuStatus();
    return info?.includes('CUDA') || info?.includes('cuda') || false;
  }

  // ============================================
  // Training Progress
  // ============================================

  /**
   * Get current epoch from progress display
   */
  async getCurrentEpoch(): Promise<string> {
    const epochDisplay = this.page.getByText(/Epoch \d+/);
    return (await epochDisplay.textContent()) || '';
  }

  /**
   * Wait for training to complete
   */
  async waitForTrainingComplete(timeout: number = 600000): Promise<void> {
    await this.wait.waitForProgressComplete({ timeout });
    await this.expectSuccessAlert(/completed|finished/i);
  }

  // ============================================
  // Models Tab
  // ============================================

  /**
   * Get list of trained models
   */
  async getTrainedModels(): Promise<Locator> {
    await this.clickModelsTab();
    return this.mainContent.locator('[data-testid="stExpander"]');
  }

  /**
   * Get trained models count
   */
  async getTrainedModelsCount(): Promise<number> {
    const models = await this.getTrainedModels();
    return await models.count();
  }

  /**
   * Expand model details
   */
  async expandModelDetails(modelName: string): Promise<void> {
    await this.clickModelsTab();
    await this.expandExpander(modelName);
  }

  // ============================================
  // TensorBoard
  // ============================================

  /**
   * Check if TensorBoard link is available
   */
  async isTensorBoardAvailable(): Promise<boolean> {
    const tbLink = this.page.getByText(/TensorBoard/i);
    return await tbLink.isVisible().catch(() => false);
  }

  // ============================================
  // Target Metrics
  // ============================================

  /**
   * Get target mAP value
   */
  async getTargetMap(): Promise<string> {
    return await this.getMetricValue('Target mAP@50');
  }

  /**
   * Get target inference time
   */
  async getTargetInference(): Promise<string> {
    return await this.getMetricValue('Target Inference');
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert training tabs are visible
   */
  async expectTrainingTabsVisible(): Promise<void> {
    await expect(this.page.getByRole('tab', { name: /Start Training/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Models/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /History/i })).toBeVisible();
  }

  /**
   * Assert configuration sub-tabs are visible
   */
  async expectConfigSubTabsVisible(): Promise<void> {
    await this.clickStartTrainingTab();
    await expect(this.page.getByRole('tab', { name: /Dataset/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Model/i })).toBeVisible();
  }

  /**
   * Assert validation warning is displayed
   */
  async expectValidationWarning(): Promise<void> {
    await this.expectWarningAlert();
  }

  /**
   * Assert training success
   */
  async expectTrainingSuccess(): Promise<void> {
    await this.expectSuccessAlert(/completed|finished|success/i);
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await this.expectTrainingTabsVisible();
  }
}
