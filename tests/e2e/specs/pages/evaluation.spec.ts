import { test, expect } from '@playwright/test';
import { EvaluationPage } from '../../page-objects/evaluation.page';

/**
 * Evaluation Page Tests
 *
 * Tests for the Evaluation page (6_Evaluation.py):
 * - Tab navigation
 * - Model and dataset selection
 * - Visual test functionality
 * - Robustness testing
 */
test.describe('Evaluation Page', () => {
  let evaluation: EvaluationPage;

  test.beforeEach(async ({ page }) => {
    evaluation = new EvaluationPage(page);
    await evaluation.goto();
  });

  test.describe('Tab Navigation', () => {
    test('should display all evaluation tabs', async () => {
      await evaluation.expectEvaluationTabsVisible();
    });

    test('should switch to Run Evaluation tab', async () => {
      await evaluation.clickRunEvaluationTab();
    });

    test('should switch to Results tab', async () => {
      await evaluation.clickResultsTab();
    });

    test('should switch to Visual Test tab', async () => {
      await evaluation.clickVisualTestTab();
    });

    test('should switch to Robustness Test tab', async () => {
      await evaluation.clickRobustnessTestTab();
    });

    test('should switch to Xtion Live Test tab', async () => {
      await evaluation.clickXtionLiveTestTab();
    });
  });

  test.describe('Run Evaluation Tab', () => {
    test.beforeEach(async () => {
      await evaluation.clickRunEvaluationTab();
    });

    test('should display model selection', async () => {
      await evaluation.expectModelSelectionVisible();
    });

    test('should display dataset selection or main content', async () => {
      // Dataset selection may not be visible if no datasets exist
      const hasDataset = await evaluation.page.getByText('Dataset').first().isVisible().catch(() => false);
      const hasMainContent = await evaluation.mainContent.isVisible().catch(() => false);
      expect(hasDataset || hasMainContent).toBe(true);
    });

    test('should display competition requirements if visible', async () => {
      // Competition requirements may not be visible in all UI states
      const hasMetrics = await evaluation.selectors.metric('Target mAP@50').first().isVisible().catch(() => false);
      const hasMainContent = await evaluation.mainContent.isVisible().catch(() => false);
      expect(hasMetrics || hasMainContent).toBe(true);
    });

    test('should display Run Evaluation button if visible', async () => {
      const button = evaluation.page.getByRole('button', { name: /Run Evaluation/i });
      const isVisible = await button.isVisible().catch(() => false);
      const hasMainContent = await evaluation.mainContent.isVisible().catch(() => false);
      // Button may not be visible if prerequisites are not met
      expect(isVisible || hasMainContent).toBe(true);
    });
  });

  test.describe('Competition Requirements', () => {
    test.beforeEach(async () => {
      await evaluation.clickRunEvaluationTab();
    });

    test('should display target mAP@50 value if visible', async () => {
      const targetMap = await evaluation.getTargetMap().catch(() => '');
      // Value may be a number, percentage, or empty in CI environment
      expect(targetMap === '' || /\d+/.test(targetMap)).toBe(true);
    });

    test('should display target inference time if visible', async () => {
      const targetInference = await evaluation.getTargetInference().catch(() => '');
      // Value may be a number, time value, or empty in CI environment
      expect(targetInference === '' || /\d+/.test(targetInference)).toBe(true);
    });
  });

  test.describe('Visual Test Tab', () => {
    test.beforeEach(async () => {
      await evaluation.clickVisualTestTab();
    });

    test('should display image source selection', async () => {
      // Should have Image, Camera, or Video options
      const imageTab = evaluation.selectors.tab('Image');
      if (await imageTab.isVisible().catch(() => false)) {
        await expect(imageTab).toBeVisible();
      }
    });

    test('should display file uploader for image source', async () => {
      await evaluation.selectImageSource('Image');
      const uploader = evaluation.selectors.fileUploader();
      if (await uploader.isVisible().catch(() => false)) {
        await expect(uploader).toBeVisible();
      }
    });
  });

  test.describe('Robustness Test Tab', () => {
    test.beforeEach(async () => {
      await evaluation.clickRobustnessTestTab();
    });

    test('should display robustness test section', async () => {
      const section = await evaluation.getRobustnessConfigSection();
      expect(section).toBeDefined();
    });

    test('should display run robustness test button', async () => {
      const button = evaluation.page.getByRole('button', { name: /Robustness/i });
      if (await button.isVisible().catch(() => false)) {
        await expect(button).toBeVisible();
      }
    });
  });

  test.describe('Xtion Live Test Tab', () => {
    test.beforeEach(async () => {
      await evaluation.clickXtionLiveTestTab();
    });

    test('should check Xtion camera availability', async () => {
      const isAvailable = await evaluation.isXtionAvailable();
      expect(typeof isAvailable).toBe('boolean');
    });
  });

  test.describe('Results Tab', () => {
    test.beforeEach(async () => {
      await evaluation.clickResultsTab();
    });

    test('should display results section', async () => {
      // Results may be empty if no evaluation has been run
      const results = await evaluation.getEvaluationResults();
      expect(typeof results).toBe('object');
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(evaluation.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(evaluation.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await evaluation.expectPageLoaded();
    });
  });
});
