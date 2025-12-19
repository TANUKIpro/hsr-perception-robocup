import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for HSR Perception E2E tests
 *
 * Streamlit-specific considerations:
 * - Single worker due to session state sharing
 * - Extended timeouts for Streamlit reruns
 * - Screenshot/video on failure for debugging
 */
export default defineConfig({
  testDir: './specs',

  // Streamlit requires sequential execution due to session state
  fullyParallel: false,
  workers: 1,

  // Retry configuration
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,

  // Reporters
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
  ],

  // Global settings
  use: {
    // Base URL for Streamlit app
    baseURL: process.env.STREAMLIT_URL || 'http://localhost:8501',

    // Trace and artifacts on failure
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',

    // Extended timeouts for Streamlit
    actionTimeout: 15000,
    navigationTimeout: 30000,

    // Browser context
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true,
  },

  // Test timeout (Streamlit can be slow)
  timeout: 60000,

  // Assertion timeout
  expect: {
    timeout: 10000,
  },

  // Browser projects
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'chromium-headless',
      use: {
        ...devices['Desktop Chrome'],
        headless: true,
      },
    },
  ],

  // Output directory for test artifacts
  outputDir: 'test-results/',
});
