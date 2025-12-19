import { Page, Locator, expect } from '@playwright/test';
import { STREAMLIT_SELECTORS } from './streamlit-selectors';

/**
 * Streamlit Wait Helpers
 *
 * Utilities for handling Streamlit's reactive behavior:
 * - Rerun cycles after state changes
 * - Loading spinners
 * - Network requests
 * - Dynamic content updates
 */
export class StreamlitWaitHelpers {
  constructor(private page: Page) {}

  /**
   * Wait for Streamlit app to complete initial load
   */
  async waitForAppLoad(options: { timeout?: number } = {}): Promise<void> {
    const timeout = options.timeout || 30000;

    // Wait for app container to be visible
    await this.page.locator(STREAMLIT_SELECTORS.APP_CONTAINER).waitFor({
      state: 'visible',
      timeout,
    });

    // Wait for initial scripts to load
    await this.page.waitForLoadState('domcontentloaded', { timeout });

    // Wait for spinners to clear
    await this.waitForSpinnerClear({ timeout: timeout / 2 });
  }

  /**
   * Wait for Streamlit rerun to complete
   *
   * Call this after any action that triggers a Streamlit rerun:
   * - Button clicks
   * - Form submissions
   * - Selectbox changes
   * - Checkbox toggles
   */
  async waitForRerun(options: { timeout?: number } = {}): Promise<void> {
    const timeout = options.timeout || 15000;

    // Brief pause to allow rerun to trigger
    await this.page.waitForTimeout(300);

    // Wait for spinner to appear and disappear
    await this.waitForSpinnerClear({ timeout });

    // Additional wait for DOM to stabilize
    await this.page.waitForTimeout(200);
  }

  /**
   * Wait for loading spinner to clear
   */
  async waitForSpinnerClear(options: { timeout?: number } = {}): Promise<void> {
    const timeout = options.timeout || 20000;
    const spinner = this.page.locator(STREAMLIT_SELECTORS.SPINNER);

    try {
      // First check if spinner appears
      await spinner.waitFor({ state: 'visible', timeout: 1000 }).catch(() => {});

      // Then wait for it to disappear
      await spinner.waitFor({ state: 'hidden', timeout });
    } catch {
      // Spinner might never appear, which is fine
    }
  }

  /**
   * Wait for network to become idle
   */
  async waitForNetworkIdle(options: { timeout?: number; minIdleTime?: number } = {}): Promise<void> {
    const timeout = options.timeout || 10000;
    const minIdleTime = options.minIdleTime || 500;

    await this.page.waitForLoadState('networkidle', { timeout });
    await this.page.waitForTimeout(minIdleTime);
  }

  /**
   * Wait for element to appear with text
   */
  async waitForText(
    text: string | RegExp,
    options: { timeout?: number; visible?: boolean } = {}
  ): Promise<Locator> {
    const timeout = options.timeout || 10000;
    const locator = this.page.getByText(text);

    await locator.waitFor({
      state: options.visible !== false ? 'visible' : 'attached',
      timeout,
    });

    return locator;
  }

  /**
   * Wait for element to disappear
   */
  async waitForElementHidden(
    selector: string,
    options: { timeout?: number } = {}
  ): Promise<void> {
    const timeout = options.timeout || 10000;
    await this.page.locator(selector).waitFor({ state: 'hidden', timeout });
  }

  /**
   * Wait for a metric to show a specific value
   */
  async waitForMetricValue(
    metricLabel: string,
    predicate: (value: string) => boolean,
    options: { timeout?: number; pollInterval?: number } = {}
  ): Promise<string> {
    const timeout = options.timeout || 30000;
    const pollInterval = options.pollInterval || 500;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const metric = this.page.locator(`${STREAMLIT_SELECTORS.METRIC}:has-text("${metricLabel}")`);
      const valueElement = metric.locator('[data-testid="stMetricValue"]');

      try {
        const value = (await valueElement.textContent({ timeout: 1000 })) || '';
        if (predicate(value)) {
          return value;
        }
      } catch {
        // Element not ready yet
      }

      await this.page.waitForTimeout(pollInterval);
    }

    throw new Error(`Metric "${metricLabel}" did not match predicate within ${timeout}ms`);
  }

  /**
   * Wait for progress bar to complete (100%)
   */
  async waitForProgressComplete(options: { timeout?: number } = {}): Promise<void> {
    const timeout = options.timeout || 60000;

    await this.page.waitForFunction(
      () => {
        const progress = document.querySelector('[data-testid="stProgress"]');
        if (!progress) return true; // No progress bar = complete

        const progressBar = progress.querySelector('[role="progressbar"]');
        const value = progressBar?.getAttribute('aria-valuenow');
        return value === '100' || value === '1';
      },
      { timeout }
    );
  }

  /**
   * Wait for toast notification to appear
   */
  async waitForToast(
    expectedText?: string | RegExp,
    options: { timeout?: number; dismiss?: boolean } = {}
  ): Promise<Locator> {
    const timeout = options.timeout || 10000;
    const toast = this.page.locator(STREAMLIT_SELECTORS.TOAST);

    await toast.waitFor({ state: 'visible', timeout });

    if (expectedText) {
      await expect(toast).toContainText(expectedText);
    }

    if (options.dismiss) {
      const closeButton = toast.locator('button[aria-label="Close"]');
      if (await closeButton.isVisible()) {
        await closeButton.click();
        await toast.waitFor({ state: 'hidden', timeout: 5000 });
      }
    }

    return toast;
  }

  /**
   * Wait for alert message to appear
   */
  async waitForAlert(
    type: 'success' | 'error' | 'warning' | 'info',
    expectedText?: string | RegExp,
    options: { timeout?: number } = {}
  ): Promise<Locator> {
    const timeout = options.timeout || 10000;
    const kindMap = {
      success: 'positive',
      error: 'negative',
      warning: 'warning',
      info: 'info',
    };

    const alert = this.page.locator(
      `[data-testid="stAlert"][data-baseweb="notification"][kind="${kindMap[type]}"]`
    );

    await alert.first().waitFor({ state: 'visible', timeout });

    if (expectedText) {
      await expect(alert.first()).toContainText(expectedText);
    }

    return alert.first();
  }

  /**
   * Wait for file upload to complete
   */
  async waitForFileUpload(options: { timeout?: number } = {}): Promise<void> {
    const timeout = options.timeout || 30000;

    // Wait for upload progress to finish
    await this.page.waitForFunction(
      () => {
        const uploader = document.querySelector('[data-testid="stFileUploader"]');
        const uploadingProgress = uploader?.querySelector('[data-testid="stProgressBar"]');
        return !uploadingProgress;
      },
      { timeout }
    );

    // Wait for rerun after upload
    await this.waitForRerun({ timeout: 5000 });
  }

  /**
   * Wait for page navigation within Streamlit
   */
  async waitForPageNavigation(
    expectedPath: string | RegExp,
    options: { timeout?: number } = {}
  ): Promise<void> {
    const timeout = options.timeout || 15000;

    await this.page.waitForURL(expectedPath, { timeout });
    await this.waitForAppLoad({ timeout });
  }

  /**
   * Poll until condition is met
   */
  async pollUntil(
    condition: () => Promise<boolean>,
    options: { timeout?: number; pollInterval?: number; message?: string } = {}
  ): Promise<void> {
    const timeout = options.timeout || 30000;
    const pollInterval = options.pollInterval || 500;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      if (await condition()) {
        return;
      }
      await this.page.waitForTimeout(pollInterval);
    }

    throw new Error(options.message || `Condition not met within ${timeout}ms`);
  }
}

/**
 * Create wait helpers instance for a page
 */
export function createWaitHelpers(page: Page): StreamlitWaitHelpers {
  return new StreamlitWaitHelpers(page);
}
