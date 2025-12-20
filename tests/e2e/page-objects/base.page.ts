import { Page, Locator, expect } from '@playwright/test';
import { StreamlitSelectors, STREAMLIT_SELECTORS } from '../utils/streamlit-selectors';
import { StreamlitWaitHelpers } from '../utils/wait-helpers';

/**
 * Base Page Object for Streamlit application
 *
 * Provides common functionality for all page objects:
 * - Navigation
 * - Streamlit-specific waiting
 * - Common UI interactions
 */
export class BasePage {
  readonly page: Page;
  readonly selectors: StreamlitSelectors;
  readonly wait: StreamlitWaitHelpers;

  // Common locators
  readonly appContainer: Locator;
  readonly sidebar: Locator;
  readonly mainContent: Locator;
  readonly spinner: Locator;

  constructor(page: Page) {
    this.page = page;
    this.selectors = new StreamlitSelectors(page);
    this.wait = new StreamlitWaitHelpers(page);

    // Initialize common locators
    this.appContainer = page.locator(STREAMLIT_SELECTORS.APP_CONTAINER);
    this.sidebar = page.locator(STREAMLIT_SELECTORS.SIDEBAR);
    this.mainContent = page.locator(STREAMLIT_SELECTORS.MAIN_CONTENT);
    this.spinner = page.locator(STREAMLIT_SELECTORS.SPINNER);
  }

  /**
   * Navigate to a path and wait for Streamlit to load
   */
  async navigate(path: string = ''): Promise<void> {
    await this.page.goto(path);
    await this.wait.waitForAppLoad();
  }

  /**
   * Click a button and wait for Streamlit rerun
   */
  async clickButton(name: string): Promise<void> {
    const button = this.page.getByRole('button', { name });
    await button.click();
    await this.wait.waitForRerun();
  }

  /**
   * Select an option from a selectbox
   */
  async selectOption(label: string, value: string): Promise<void> {
    const selectbox = this.selectors.selectbox(label);
    await selectbox.click();

    // Wait for dropdown menu to appear
    const option = this.page.locator(`[data-baseweb="menu"] >> text="${value}"`);
    await option.waitFor({ state: 'visible' });
    await option.click();

    await this.wait.waitForRerun();
  }

  /**
   * Fill a text input
   */
  async fillTextInput(label: string, value: string): Promise<void> {
    const input = this.selectors.textInput(label);
    await input.clear();
    await input.fill(value);
    await this.wait.waitForRerun();
  }

  /**
   * Fill a number input
   */
  async fillNumberInput(label: string, value: number): Promise<void> {
    const input = this.selectors.numberInput(label);
    await input.clear();
    await input.fill(String(value));
    await this.wait.waitForRerun();
  }

  /**
   * Toggle a checkbox
   */
  async toggleCheckbox(label: string): Promise<void> {
    const checkbox = this.selectors.checkbox(label);
    await checkbox.click();
    await this.wait.waitForRerun();
  }

  /**
   * Select a radio option
   */
  async selectRadio(groupLabel: string, optionLabel: string): Promise<void> {
    const radio = this.selectors.radio(groupLabel);
    const option = radio.getByText(optionLabel);
    await option.click();
    await this.wait.waitForRerun();
  }

  /**
   * Click a tab (partial match, case-insensitive)
   */
  async clickTab(tabName: string): Promise<void> {
    const tab = this.selectors.tab(tabName);
    await tab.click();
    await this.wait.waitForRerun();
  }

  /**
   * Click a tab by exact text (including emojis)
   */
  async clickTabExact(tabName: string): Promise<void> {
    const tab = this.selectors.tabExact(tabName);
    await tab.click();
    await this.wait.waitForRerun();
  }

  /**
   * Expand an expander
   */
  async expandExpander(title: string): Promise<void> {
    const expander = this.selectors.expander(title);
    const isExpanded = await expander.getAttribute('aria-expanded');
    if (isExpanded !== 'true') {
      await expander.click();
      await this.wait.waitForRerun();
    }
  }

  /**
   * Upload a file
   */
  async uploadFile(filePath: string, label?: string): Promise<void> {
    const uploader = this.selectors.fileUploader(label);
    const input = uploader.locator('input[type="file"]');
    await input.setInputFiles(filePath);
    await this.wait.waitForFileUpload();
  }

  /**
   * Get metric value
   */
  async getMetricValue(label: string): Promise<string> {
    // Use .first() to avoid strict mode violation when multiple metrics match
    const metricValue = this.selectors.metricValue(label).first();
    return (await metricValue.textContent()) || '';
  }

  /**
   * Assert success alert is visible
   */
  async expectSuccessAlert(text?: string | RegExp): Promise<void> {
    const alert = await this.wait.waitForAlert('success', text);
    await expect(alert).toBeVisible();
  }

  /**
   * Assert error alert is visible
   */
  async expectErrorAlert(text?: string | RegExp): Promise<void> {
    const alert = await this.wait.waitForAlert('error', text);
    await expect(alert).toBeVisible();
  }

  /**
   * Assert warning alert is visible
   */
  async expectWarningAlert(text?: string | RegExp): Promise<void> {
    const alert = await this.wait.waitForAlert('warning', text);
    await expect(alert).toBeVisible();
  }

  /**
   * Assert info alert is visible
   */
  async expectInfoAlert(text?: string | RegExp): Promise<void> {
    const alert = await this.wait.waitForAlert('info', text);
    await expect(alert).toBeVisible();
  }

  /**
   * Assert toast notification is visible
   */
  async expectToast(text?: string | RegExp): Promise<void> {
    await this.wait.waitForToast(text);
  }

  /**
   * Get current page URL
   */
  get currentUrl(): string {
    return this.page.url();
  }

  /**
   * Get page title
   */
  async getPageTitle(): Promise<string> {
    return await this.page.title();
  }

  /**
   * Check if element with text exists
   */
  async hasText(text: string | RegExp): Promise<boolean> {
    const element = this.page.getByText(text);
    return await element.isVisible().catch(() => false);
  }

  /**
   * Get all visible text content from main area
   */
  async getMainContent(): Promise<string> {
    return (await this.mainContent.textContent()) || '';
  }

  /**
   * Take a screenshot
   */
  async screenshot(name: string): Promise<void> {
    await this.page.screenshot({
      path: `test-results/screenshots/${name}.png`,
      fullPage: true,
    });
  }
}
