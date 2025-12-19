import { Page, Locator, expect } from '@playwright/test';
import { StreamlitSelectors, STREAMLIT_SELECTORS } from '../utils/streamlit-selectors';
import { StreamlitWaitHelpers } from '../utils/wait-helpers';

/**
 * Sidebar Component Object
 *
 * Represents the common sidebar present on all pages:
 * - Profile selector
 * - Navigation links
 * - Registry statistics
 * - Active task status
 */
export class SidebarComponent {
  readonly page: Page;
  readonly selectors: StreamlitSelectors;
  readonly wait: StreamlitWaitHelpers;
  readonly container: Locator;

  constructor(page: Page) {
    this.page = page;
    this.selectors = new StreamlitSelectors(page);
    this.wait = new StreamlitWaitHelpers(page);
    this.container = page.locator(STREAMLIT_SELECTORS.SIDEBAR);
  }

  // ============================================
  // Navigation
  // ============================================

  /**
   * Navigate to a page using sidebar link
   */
  async navigateTo(pageName: string): Promise<void> {
    // Try different selector strategies for Streamlit navigation
    const link = this.container.locator(`a:has-text("${pageName}")`).first();
    await link.click();
    await this.wait.waitForRerun();
  }

  /**
   * Navigate to Home page
   */
  async goToHome(): Promise<void> {
    await this.navigateTo('Home');
  }

  /**
   * Navigate to Dashboard page
   */
  async goToDashboard(): Promise<void> {
    await this.navigateTo('Dashboard');
  }

  /**
   * Navigate to Registry page
   */
  async goToRegistry(): Promise<void> {
    await this.navigateTo('Registry');
  }

  /**
   * Navigate to Collection page
   */
  async goToCollection(): Promise<void> {
    await this.navigateTo('Collection');
  }

  /**
   * Navigate to Annotation page
   */
  async goToAnnotation(): Promise<void> {
    await this.navigateTo('Annotation');
  }

  /**
   * Navigate to Training page
   */
  async goToTraining(): Promise<void> {
    await this.navigateTo('Training');
  }

  /**
   * Navigate to Evaluation page
   */
  async goToEvaluation(): Promise<void> {
    await this.navigateTo('Evaluation');
  }

  /**
   * Navigate to Settings page
   */
  async goToSettings(): Promise<void> {
    await this.navigateTo('Settings');
  }

  /**
   * Get all navigation links
   */
  async getNavigationLinks(): Promise<string[]> {
    const links = this.container.locator('a');
    const count = await links.count();
    const linkTexts: string[] = [];

    for (let i = 0; i < count; i++) {
      const text = await links.nth(i).textContent();
      if (text) {
        linkTexts.push(text.trim());
      }
    }

    return linkTexts;
  }

  // ============================================
  // Profile Management
  // ============================================

  /**
   * Get current profile name from selector
   */
  async getCurrentProfileName(): Promise<string> {
    const profileSelector = this.container.locator('[data-testid="stSelectbox"]').first();
    // Try multiple strategies for getting selected value
    // Strategy 1: Get text directly from the selectbox div that shows current value
    const selectContainer = profileSelector.locator('[data-baseweb="select"]').first();

    // The selected value is typically in a div with the displayed text
    // Different Streamlit/Baseweb versions may have different structures
    const displayedValue = selectContainer.locator('div[aria-selected="true"], div.css-1hwfws3, span:not([class*="arrow"])').first();

    try {
      await displayedValue.waitFor({ state: 'visible', timeout: 3000 });
      const text = await displayedValue.textContent();
      return text?.trim() || '';
    } catch {
      // Fallback: get the inner text of the select container
      const containerText = await selectContainer.innerText();
      // Return the first non-empty line (typically the selected value)
      const lines = containerText.split('\n').filter(l => l.trim());
      return lines[0]?.trim() || '';
    }
  }

  /**
   * Select a profile by name
   */
  async selectProfile(profileName: string): Promise<void> {
    const profileSelector = this.container.locator('[data-testid="stSelectbox"]').first();
    await profileSelector.click();

    const option = this.page.locator(`[data-baseweb="menu"] >> text="${profileName}"`);
    await option.waitFor({ state: 'visible' });
    await option.click();

    await this.wait.waitForRerun();
  }

  /**
   * Check if profile selector is visible
   */
  async isProfileSelectorVisible(): Promise<boolean> {
    const selector = this.container.locator('[data-testid="stSelectbox"]').first();
    return await selector.isVisible().catch(() => false);
  }

  // ============================================
  // Statistics & Status
  // ============================================

  /**
   * Get registry statistics displayed in sidebar
   */
  async getRegistryStats(): Promise<{
    totalObjects?: string;
    totalImages?: string;
  }> {
    const stats: { totalObjects?: string; totalImages?: string } = {};

    // Look for metrics in sidebar
    const metrics = this.container.locator('[data-testid="stMetric"]');
    const count = await metrics.count();

    for (let i = 0; i < count; i++) {
      const metric = metrics.nth(i);
      const label = await metric.locator('[data-testid="stMetricLabel"]').textContent();
      const value = await metric.locator('[data-testid="stMetricValue"]').textContent();

      if (label?.includes('Object')) {
        stats.totalObjects = value || undefined;
      } else if (label?.includes('Image')) {
        stats.totalImages = value || undefined;
      }
    }

    return stats;
  }

  /**
   * Check if there's an active task indicator
   */
  async hasActiveTask(): Promise<boolean> {
    // Look for active task indicators (usually a spinner or status text)
    const activeIndicator = this.container.locator('text=/Running|In Progress|Active/i');
    return await activeIndicator.isVisible().catch(() => false);
  }

  /**
   * Get active task status text
   */
  async getActiveTaskStatus(): Promise<string | null> {
    const statusText = this.container.locator('text=/Running|In Progress|Active/i').first();
    if (await statusText.isVisible().catch(() => false)) {
      return await statusText.textContent();
    }
    return null;
  }

  // ============================================
  // Visibility & State
  // ============================================

  /**
   * Check if sidebar is visible
   */
  async isVisible(): Promise<boolean> {
    return await this.container.isVisible();
  }

  /**
   * Wait for sidebar to load completely
   */
  async waitForLoad(): Promise<void> {
    await this.container.waitFor({ state: 'visible' });
    await this.wait.waitForSpinnerClear();
  }

  /**
   * Get sidebar width (useful for responsive testing)
   */
  async getWidth(): Promise<number> {
    const box = await this.container.boundingBox();
    return box?.width || 0;
  }

  /**
   * Collapse sidebar (if supported by the app)
   */
  async collapse(): Promise<void> {
    const collapseButton = this.container.locator('button[aria-label*="Collapse"]');
    if (await collapseButton.isVisible()) {
      await collapseButton.click();
      await this.wait.waitForRerun();
    }
  }

  /**
   * Expand sidebar (if supported by the app)
   */
  async expand(): Promise<void> {
    const expandButton = this.page.locator('button[aria-label*="Expand"]');
    if (await expandButton.isVisible()) {
      await expandButton.click();
      await this.wait.waitForRerun();
    }
  }
}
