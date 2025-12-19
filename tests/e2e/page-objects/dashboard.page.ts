import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Dashboard Page Object
 *
 * Page: 1_Dashboard.py
 * Features:
 * - Overall statistics (total objects, images, target metrics)
 * - Pipeline status
 * - Category progress breakdown
 * - Training readiness checker
 */
export class DashboardPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Dashboard';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Dashboard page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Statistics Section
  // ============================================

  /**
   * Get total objects count
   */
  async getTotalObjects(): Promise<string> {
    return await this.getMetricValue('Total Objects');
  }

  /**
   * Get images collected count
   */
  async getImagesCollected(): Promise<string> {
    return await this.getMetricValue('Images Collected');
  }

  /**
   * Get target total
   */
  async getTargetTotal(): Promise<string> {
    return await this.getMetricValue('Target Total');
  }

  /**
   * Get ready for training count
   */
  async getReadyForTraining(): Promise<string> {
    return await this.getMetricValue('Ready for Training');
  }

  // ============================================
  // Pipeline Status Section
  // ============================================

  /**
   * Get annotated datasets count
   */
  async getAnnotatedDatasets(): Promise<string> {
    return await this.getMetricValue('Annotated Datasets');
  }

  /**
   * Get trained models count
   */
  async getTrainedModels(): Promise<string> {
    return await this.getMetricValue('Trained Models');
  }

  /**
   * Get active tasks count
   */
  async getActiveTasks(): Promise<string> {
    return await this.getMetricValue('Active Tasks');
  }

  // ============================================
  // Category Progress Section
  // ============================================

  /**
   * Check if category progress section is visible
   */
  async isCategoryProgressVisible(): Promise<boolean> {
    const section = this.page.getByText('Progress by Category');
    return await section.isVisible().catch(() => false);
  }

  /**
   * Get progress bars for all categories
   */
  async getCategoryProgressBars(): Promise<Locator> {
    return this.mainContent.locator('[data-testid="stProgress"]');
  }

  // ============================================
  // Training Readiness Section
  // ============================================

  /**
   * Check if training readiness section is visible
   */
  async isTrainingReadinessVisible(): Promise<boolean> {
    const section = this.page.getByText('Training Readiness');
    return await section.isVisible().catch(() => false);
  }

  /**
   * Check if export button is enabled
   */
  async isExportButtonEnabled(): Promise<boolean> {
    const button = this.page.getByRole('button', { name: /Export to YOLO/i });
    if (await button.isVisible().catch(() => false)) {
      return await button.isEnabled();
    }
    return false;
  }

  /**
   * Click export to YOLO config button
   */
  async exportToYoloConfig(): Promise<void> {
    await this.clickButton('Export to YOLO Config');
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert statistics section is displayed
   */
  async expectStatisticsDisplayed(): Promise<void> {
    // Use .first() to avoid strict mode violation when multiple metrics match
    await expect(this.selectors.metric('Total Objects').first()).toBeVisible();
    await expect(this.selectors.metric('Images Collected').first()).toBeVisible();
  }

  /**
   * Assert pipeline status is displayed
   */
  async expectPipelineStatusDisplayed(): Promise<void> {
    await expect(this.page.getByText('Pipeline Status')).toBeVisible();
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await this.expectStatisticsDisplayed();
  }
}
