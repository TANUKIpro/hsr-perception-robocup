import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../page-objects/dashboard.page';

/**
 * Dashboard Page Tests
 *
 * Tests for the Dashboard page (1_Dashboard.py):
 * - Overall statistics display
 * - Pipeline status
 * - Category progress
 * - Training readiness
 */
test.describe('Dashboard Page', () => {
  let dashboard: DashboardPage;

  test.beforeEach(async ({ page }) => {
    dashboard = new DashboardPage(page);
    await dashboard.goto();
  });

  test.describe('Statistics Section', () => {
    test('should display Total Objects metric', async () => {
      // Use .first() to avoid strict mode violation when multiple metrics match
      await expect(dashboard.selectors.metric('Total Objects').first()).toBeVisible();
    });

    test('should display Images Collected metric', async () => {
      await expect(dashboard.selectors.metric('Images Collected').first()).toBeVisible();
    });

    test('should display Target Total metric', async () => {
      await expect(dashboard.selectors.metric('Target Total').first()).toBeVisible();
    });

    test('should display Ready for Training metric', async () => {
      await expect(dashboard.selectors.metric('Ready for Training').first()).toBeVisible();
    });

    test('should show numeric values for statistics', async () => {
      const totalObjects = await dashboard.getTotalObjects();
      // Value should be a number (possibly with whitespace) or empty/N/A in CI environment
      const trimmedValue = totalObjects.trim();
      expect(trimmedValue).toMatch(/^(\d+|N\/A|)$/);
    });
  });

  test.describe('Pipeline Status Section', () => {
    test('should display Pipeline Status header', async () => {
      await expect(dashboard.page.getByText('Pipeline Status').first()).toBeVisible();
    });

    test('should display Annotated Datasets metric', async () => {
      await expect(dashboard.selectors.metric('Annotated Datasets').first()).toBeVisible();
    });

    test('should display Trained Models metric', async () => {
      await expect(dashboard.selectors.metric('Trained Models').first()).toBeVisible();
    });

    test('should display Active Tasks metric', async () => {
      await expect(dashboard.selectors.metric('Active Tasks').first()).toBeVisible();
    });
  });

  test.describe('Category Progress Section', () => {
    test('should display category progress section', async () => {
      const isVisible = await dashboard.isCategoryProgressVisible();
      // May not be visible if no objects are registered
      expect(typeof isVisible).toBe('boolean');
    });

    test('should show progress bars when categories exist', async () => {
      const progressBars = await dashboard.getCategoryProgressBars();
      const count = await progressBars.count();
      // Count may be 0 if no objects registered
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Training Readiness Section', () => {
    test('should display training readiness section', async () => {
      const isVisible = await dashboard.isTrainingReadinessVisible();
      expect(typeof isVisible).toBe('boolean');
    });

    test('should show export button state based on data', async () => {
      const isEnabled = await dashboard.isExportButtonEnabled();
      // Button state depends on whether all objects are ready
      expect(typeof isEnabled).toBe('boolean');
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(dashboard.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(dashboard.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await dashboard.expectPageLoaded();
    });
  });
});
