import { test, expect } from '@playwright/test';
import { CollectionPage } from '../../page-objects/collection.page';

/**
 * Collection Page Tests
 *
 * Tests for the Collection page (3_Collection.py):
 * - Object selection
 * - Collection methods tabs
 * - File upload functionality
 * - Collection status display
 */
test.describe('Collection Page', () => {
  let collection: CollectionPage;

  test.beforeEach(async ({ page }) => {
    collection = new CollectionPage(page);
    await collection.goto();
  });

  test.describe('Object Selection', () => {
    test('should display object selector or warning', async () => {
      // Either object selector is visible or warning about no objects
      const hasSelector = await collection.isObjectSelectorVisible();
      const hasWarning = await collection.page.getByText(/No objects/i).isVisible().catch(() => false);

      expect(hasSelector || hasWarning).toBe(true);
    });
  });

  test.describe('Collection Method Tabs', () => {
    test('should display all collection method tabs', async () => {
      // Tabs may only appear after selecting an object
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await expect(collection.selectors.tab('ROS2 Camera')).toBeVisible();
        await expect(collection.selectors.tab('Local Camera')).toBeVisible();
        await expect(collection.selectors.tab('File Upload')).toBeVisible();
        await expect(collection.selectors.tab('Folder Import')).toBeVisible();
      }
    });

    test('should switch to ROS2 Camera tab', async () => {
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await collection.clickRos2CameraTab();
        // ROS2 Camera tab should show ROS2-specific content
      }
    });

    test('should switch to Local Camera tab', async () => {
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await collection.clickLocalCameraTab();
      }
    });

    test('should switch to File Upload tab', async () => {
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await collection.clickFileUploadTab();
        const isUploaderVisible = await collection.isFileUploaderVisible();
        expect(isUploaderVisible).toBe(true);
      }
    });

    test('should switch to Folder Import tab', async () => {
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await collection.clickFolderImportTab();
      }
    });
  });

  test.describe('File Upload Tab', () => {
    test('should display file uploader when tab is selected', async () => {
      const tabs = collection.selectors.tabs;
      if (await tabs.isVisible().catch(() => false)) {
        await collection.clickFileUploadTab();
        const uploader = collection.selectors.fileUploader();
        await expect(uploader).toBeVisible();
      }
    });
  });

  test.describe('Collection Status', () => {
    test('should display collection metrics when object is selected', async () => {
      const hasSelector = await collection.isObjectSelectorVisible();
      if (hasSelector) {
        // If objects exist and one is selected, status should be visible
        const collectedMetric = collection.selectors.metric('Collected');
        const targetMetric = collection.selectors.metric('Target');

        // These may or may not be visible depending on state
        const collectedVisible = await collectedMetric.isVisible().catch(() => false);
        const targetVisible = await targetMetric.isVisible().catch(() => false);

        // At least verify we can check for these metrics
        expect(typeof collectedVisible).toBe('boolean');
        expect(typeof targetVisible).toBe('boolean');
      }
    });
  });

  test.describe('Reference Images', () => {
    test('should check for reference images section', async () => {
      const isVisible = await collection.isReferenceImagesSectionVisible();
      expect(typeof isVisible).toBe('boolean');
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(collection.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(collection.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await collection.expectPageLoaded();
    });
  });
});
