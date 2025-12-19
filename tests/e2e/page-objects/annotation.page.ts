import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Annotation Page Object
 *
 * Page: 4_Annotation.py
 * Features:
 * - SAM2 annotation orchestration
 * - Background selection
 * - Annotation session creation
 * - Progress tracking
 */
export class AnnotationPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Annotation';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Annotation page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Tabs
  // ============================================

  /**
   * Click Run Annotation tab
   */
  async clickRunAnnotationTab(): Promise<void> {
    await this.clickTabExact('🎯 Run Annotation');
  }

  /**
   * Click Prepare Dataset tab
   */
  async clickPrepareDatasetTab(): Promise<void> {
    await this.clickTabExact('📦 Prepare Dataset');
  }

  /**
   * Click Sessions tab
   */
  async clickSessionsTab(): Promise<void> {
    await this.clickTabExact('📁 Sessions');
  }

  /**
   * Click History tab
   */
  async clickHistoryTab(): Promise<void> {
    await this.clickTabExact('📜 History');
  }

  // ============================================
  // Run Annotation Tab
  // ============================================

  /**
   * Select class to annotate
   */
  async selectClassToAnnotate(className: string): Promise<void> {
    await this.selectOption('Select Class', className);
  }

  /**
   * Select background image
   */
  async selectBackground(backgroundName: string): Promise<void> {
    await this.selectOption('Background', backgroundName);
  }

  /**
   * Select device (cuda/cpu)
   */
  async selectDevice(device: 'cuda' | 'cpu'): Promise<void> {
    await this.selectRadio('Device', device);
  }

  /**
   * Click start annotation button
   */
  async startAnnotation(): Promise<void> {
    await this.clickButton('Start Annotation');
  }

  /**
   * Check if start button is enabled
   */
  async isStartButtonEnabled(): Promise<boolean> {
    const button = this.page.getByRole('button', { name: 'Start Annotation' });
    if (await button.isVisible().catch(() => false)) {
      return await button.isEnabled();
    }
    return false;
  }

  // ============================================
  // Prepare Dataset Tab
  // ============================================

  /**
   * Get class status information
   */
  async getClassStatusSection(): Promise<Locator> {
    return this.page.getByText('Class Status').locator('..');
  }

  /**
   * Select classes for dataset preparation
   */
  async selectClassesForDataset(classNames: string[]): Promise<void> {
    const multiselect = this.selectors.multiselect('Select Classes');
    await multiselect.click();

    for (const className of classNames) {
      const option = this.page.locator(`[data-baseweb="menu"] >> text="${className}"`);
      await option.click();
    }

    // Close dropdown by clicking outside
    await this.mainContent.click({ position: { x: 10, y: 10 } });
    await this.wait.waitForRerun();
  }

  /**
   * Click prepare dataset button
   */
  async prepareDataset(): Promise<void> {
    await this.clickButton('Prepare Dataset');
  }

  // ============================================
  // Sessions Tab
  // ============================================

  /**
   * Get list of annotation sessions
   */
  async getSessionsList(): Promise<Locator> {
    return this.mainContent.locator('[data-testid="stExpander"]');
  }

  /**
   * Get sessions count
   */
  async getSessionsCount(): Promise<number> {
    const sessions = await this.getSessionsList();
    return await sessions.count();
  }

  /**
   * Expand a session by name/date
   */
  async expandSession(sessionIdentifier: string): Promise<void> {
    await this.expandExpander(sessionIdentifier);
  }

  // ============================================
  // History Tab
  // ============================================

  /**
   * Get annotation history entries
   */
  async getHistoryEntries(): Promise<Locator> {
    return this.mainContent.locator('[data-testid="stExpander"]');
  }

  // ============================================
  // Progress Monitoring
  // ============================================

  /**
   * Check if annotation is in progress
   */
  async isAnnotationInProgress(): Promise<boolean> {
    const progressBar = this.selectors.progressBar;
    return await progressBar.isVisible().catch(() => false);
  }

  /**
   * Wait for annotation to complete
   */
  async waitForAnnotationComplete(timeout: number = 300000): Promise<void> {
    await this.wait.waitForProgressComplete({ timeout });
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert annotation tabs are visible
   */
  async expectAnnotationTabsVisible(): Promise<void> {
    await expect(this.page.getByRole('tab', { name: /Run Annotation/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Prepare Dataset/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Sessions/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /History/i })).toBeVisible();
  }

  /**
   * Assert class selection is visible
   */
  async expectClassSelectionVisible(): Promise<void> {
    await expect(this.page.getByText('Select Class')).toBeVisible();
  }

  /**
   * Assert device selection is visible
   */
  async expectDeviceSelectionVisible(): Promise<void> {
    await expect(this.selectors.radio('Device')).toBeVisible();
  }

  /**
   * Assert annotation success
   */
  async expectAnnotationSuccess(): Promise<void> {
    await this.expectSuccessAlert(/completed|finished/i);
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await this.expectAnnotationTabsVisible();
  }
}
