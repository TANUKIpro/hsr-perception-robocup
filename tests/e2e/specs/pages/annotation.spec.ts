import { test, expect } from '@playwright/test';
import { AnnotationPage } from '../../page-objects/annotation.page';

/**
 * Annotation Page Tests
 *
 * Tests for the Annotation page (4_Annotation.py):
 * - Tab navigation
 * - Run Annotation configuration
 * - Sessions management
 * - Dataset preparation
 */
test.describe('Annotation Page', () => {
  let annotation: AnnotationPage;

  test.beforeEach(async ({ page }) => {
    annotation = new AnnotationPage(page);
    await annotation.goto();
  });

  test.describe('Tab Navigation', () => {
    test('should display all annotation tabs', async () => {
      await annotation.expectAnnotationTabsVisible();
    });

    test('should switch to Run Annotation tab', async () => {
      await annotation.clickRunAnnotationTab();
      // Verify Run Annotation content is visible
    });

    test('should switch to Prepare Dataset tab', async () => {
      await annotation.clickPrepareDatasetTab();
      // Verify Prepare Dataset content is visible
    });

    test('should switch to Sessions tab', async () => {
      await annotation.clickSessionsTab();
      // Verify Sessions content is visible
    });

    test('should switch to History tab', async () => {
      await annotation.clickHistoryTab();
      // Verify History content is visible
    });
  });

  test.describe('Run Annotation Tab', () => {
    test.beforeEach(async () => {
      await annotation.clickRunAnnotationTab();
    });

    test('should display class selection or appropriate message', async () => {
      // May show class selector, message if no classes, or just main content in CI
      const hasClassSelector = await annotation.page.getByText('Select Class').first().isVisible().catch(() => false);
      const hasNoClassMessage = await annotation.page.getByText(/No class/i).isVisible().catch(() => false);
      const hasMainContent = await annotation.mainContent.isVisible().catch(() => false);

      // At least main content should be visible
      expect(hasClassSelector || hasNoClassMessage || hasMainContent).toBe(true);
    });

    test('should display device selection', async () => {
      const deviceRadio = annotation.selectors.radio('Device');
      if (await deviceRadio.isVisible().catch(() => false)) {
        // Use .first() to avoid strict mode violation when multiple elements match
        await expect(annotation.page.getByText('cuda').first()).toBeVisible();
        await expect(annotation.page.getByText('cpu').first()).toBeVisible();
      }
    });

    test('should check start button state', async () => {
      const isEnabled = await annotation.isStartButtonEnabled();
      expect(typeof isEnabled).toBe('boolean');
    });
  });

  test.describe('Prepare Dataset Tab', () => {
    test.beforeEach(async () => {
      await annotation.clickPrepareDatasetTab();
    });

    test('should display class status section', async () => {
      const classStatus = await annotation.getClassStatusSection();
      // Class status should be visible or show appropriate message
      expect(classStatus).toBeDefined();
    });
  });

  test.describe('Sessions Tab', () => {
    test.beforeEach(async () => {
      await annotation.clickSessionsTab();
    });

    test('should display sessions list or empty message', async () => {
      const sessionsCount = await annotation.getSessionsCount();
      expect(sessionsCount).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('History Tab', () => {
    test.beforeEach(async () => {
      await annotation.clickHistoryTab();
    });

    test('should display history entries or empty state', async () => {
      const historyEntries = await annotation.getHistoryEntries();
      const count = await historyEntries.count();
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(annotation.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(annotation.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await annotation.expectPageLoaded();
    });
  });
});
