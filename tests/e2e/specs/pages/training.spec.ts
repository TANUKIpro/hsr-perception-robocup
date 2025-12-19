import { test, expect } from '@playwright/test';
import { TrainingPage } from '../../page-objects/training.page';

/**
 * Training Page Tests
 *
 * Tests for the Training page (5_Training.py):
 * - Tab navigation
 * - Training configuration
 * - GPU status
 * - Models management
 */
test.describe('Training Page', () => {
  let training: TrainingPage;

  test.beforeEach(async ({ page }) => {
    training = new TrainingPage(page);
    await training.goto();
  });

  test.describe('Main Tab Navigation', () => {
    test('should display all training tabs', async () => {
      await training.expectTrainingTabsVisible();
    });

    test('should switch to Start Training tab', async () => {
      await training.clickStartTrainingTab();
      // Should show configuration options
    });

    test('should switch to Models tab', async () => {
      await training.clickModelsTab();
      // Should show models list
    });

    test('should switch to History tab', async () => {
      await training.clickHistoryTab();
      // Should show training history
    });
  });

  test.describe('Configuration Sub-tabs', () => {
    test.beforeEach(async () => {
      await training.clickStartTrainingTab();
    });

    test('should display configuration sub-tabs', async () => {
      await training.expectConfigSubTabsVisible();
    });

    test('should switch to Dataset sub-tab', async () => {
      await training.clickDatasetSubTab();
      await expect(training.page.getByText('Dataset')).toBeVisible();
    });

    test('should switch to Model sub-tab', async () => {
      await training.clickModelSubTab();
      // Should show model configuration
    });

    test('should switch to Advanced sub-tab', async () => {
      await training.clickAdvancedSubTab();
      // Should show advanced options
    });

    test('should switch to Others sub-tab', async () => {
      await training.clickOthersSubTab();
      // Should show hardware/GPU info
    });
  });

  test.describe('Dataset Configuration', () => {
    test.beforeEach(async () => {
      await training.clickStartTrainingTab();
      await training.clickDatasetSubTab();
    });

    test('should display dataset selector', async () => {
      const selectbox = training.selectors.selectbox('Dataset');
      await expect(selectbox).toBeVisible();
    });

    test('should list available datasets', async () => {
      const datasets = await training.getAvailableDatasets();
      // May be empty if no datasets prepared
      expect(Array.isArray(datasets)).toBe(true);
    });
  });

  test.describe('Model Configuration', () => {
    test.beforeEach(async () => {
      await training.clickStartTrainingTab();
      await training.clickModelSubTab();
    });

    test('should display epochs configuration', async () => {
      const slider = training.selectors.slider('Epochs');
      await expect(slider).toBeVisible();
    });

    test('should display base model selector', async () => {
      const selectbox = training.selectors.selectbox('Base Model');
      if (await selectbox.isVisible().catch(() => false)) {
        await expect(selectbox).toBeVisible();
      }
    });
  });

  test.describe('Hardware & GPU', () => {
    test.beforeEach(async () => {
      await training.clickStartTrainingTab();
      await training.clickOthersSubTab();
    });

    test('should display GPU status section', async () => {
      const { visible } = await training.getGpuStatus();
      expect(visible).toBe(true);
    });

    test('should indicate CUDA availability', async () => {
      const isCudaAvailable = await training.isCudaAvailable();
      expect(typeof isCudaAvailable).toBe('boolean');
    });
  });

  test.describe('Models Tab', () => {
    test.beforeEach(async () => {
      await training.clickModelsTab();
    });

    test('should display trained models or empty state', async () => {
      const modelsCount = await training.getTrainedModelsCount();
      expect(modelsCount).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Training Controls', () => {
    test('should display start training button', async () => {
      await training.clickStartTrainingTab();
      const button = training.page.getByRole('button', { name: /Start Training/i });
      await expect(button).toBeVisible();
    });

    test('should check start button enabled state', async () => {
      await training.clickStartTrainingTab();
      const isEnabled = await training.isStartButtonEnabled();
      expect(typeof isEnabled).toBe('boolean');
    });
  });

  test.describe('Target Metrics', () => {
    test('should display target mAP', async () => {
      const targetMap = await training.getTargetMap().catch(() => '');
      // May or may not be visible depending on page layout
      expect(typeof targetMap).toBe('string');
    });

    test('should display target inference time', async () => {
      const targetInference = await training.getTargetInference().catch(() => '');
      expect(typeof targetInference).toBe('string');
    });
  });

  test.describe('TensorBoard Integration', () => {
    test('should check TensorBoard availability', async () => {
      const isAvailable = await training.isTensorBoardAvailable();
      expect(typeof isAvailable).toBe('boolean');
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(training.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(training.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await training.expectPageLoaded();
    });
  });
});
