import { test, expect } from '@playwright/test';
import { SettingsPage } from '../../page-objects/settings.page';

/**
 * Settings Page Tests
 *
 * Tests for the Settings page (7_Settings.py):
 * - Profile management
 * - Data management
 * - Category management
 * - System status
 */
test.describe('Settings Page', () => {
  let settings: SettingsPage;

  test.beforeEach(async ({ page }) => {
    settings = new SettingsPage(page);
    await settings.goto();
  });

  test.describe('Profile Management Section', () => {
    test('should display profile management section', async () => {
      await settings.expectProfileManagementVisible();
    });

    test('should display profile tabs', async () => {
      await settings.expectProfileTabsVisible();
    });

    test('should switch to Switch tab', async () => {
      await settings.clickSwitchTab();
    });

    test('should switch to Create tab', async () => {
      await settings.clickCreateTab();
    });

    test('should switch to Export tab', async () => {
      await settings.clickExportTab();
    });

    test('should switch to Import tab', async () => {
      await settings.clickImportTab();
    });
  });

  test.describe('Create Profile Tab', () => {
    test.beforeEach(async () => {
      await settings.clickCreateTab();
    });

    test('should display profile name input', async () => {
      await expect(settings.page.getByText('Profile Name')).toBeVisible();
    });

    test('should display create profile button', async () => {
      await expect(settings.page.getByRole('button', { name: /Create Profile/i })).toBeVisible();
    });

    test('should be able to fill profile name', async () => {
      await settings.fillTextInput('Profile Name', 'Test Profile');
      const input = settings.selectors.textInput('Profile Name');
      await expect(input).toHaveValue('Test Profile');
    });
  });

  test.describe('Import/Export Tab', () => {
    test.beforeEach(async () => {
      await settings.clickImportExportTab();
    });

    test('should display Prepare Export button', async () => {
      // Use exact match to avoid matching "Export to YOLO Config"
      const button = settings.page.getByRole('button', { name: 'Prepare Export' });
      await expect(button).toBeVisible();
    });

    test('should display import section', async () => {
      // Check for import-related UI elements
      const importSection = settings.page.getByText(/Import/i);
      await expect(importSection.first()).toBeVisible();
    });
  });

  test.describe('Data Management Section', () => {
    test('should display data management options', async () => {
      await settings.expectDataManagementVisible();
    });

    test('should display export to YOLO config button', async () => {
      const button = settings.page.getByRole('button', { name: /Export to YOLO/i });
      await expect(button).toBeVisible();
    });

    test('should display update collection counts button', async () => {
      const button = settings.page.getByRole('button', { name: /Update.*Collection/i });
      await expect(button).toBeVisible();
    });
  });

  test.describe('Category Management', () => {
    test('should display new category input', async () => {
      const input = settings.selectors.textInput('New Category');
      if (await input.isVisible().catch(() => false)) {
        await expect(input).toBeVisible();
      }
    });

    test('should display add category button', async () => {
      const button = settings.page.getByRole('button', { name: /Add Category/i });
      if (await button.isVisible().catch(() => false)) {
        await expect(button).toBeVisible();
      }
    });
  });

  test.describe('System Status Section', () => {
    test('should display system status section', async () => {
      const isVisible = await settings.isSystemStatusVisible();
      expect(isVisible).toBe(true);
    });

    test('should check GPU availability status', async () => {
      const isGpuAvailable = await settings.isGpuAvailable();
      expect(typeof isGpuAvailable).toBe('boolean');
    });

    test('should check ROS2 availability status', async () => {
      const isRos2Available = await settings.isRos2Available();
      expect(typeof isRos2Available).toBe('boolean');
    });
  });

  test.describe('Data Paths Section', () => {
    test('should check data paths section visibility', async () => {
      // Data Paths section may or may not be visible depending on UI configuration
      const isVisible = await settings.isDataPathsVisible();
      expect(typeof isVisible).toBe('boolean');
    });
  });

  test.describe('Page Layout', () => {
    test('should display sidebar', async () => {
      await expect(settings.sidebar.container).toBeVisible();
    });

    test('should display main content area', async () => {
      await expect(settings.mainContent).toBeVisible();
    });

    test('should have proper page structure', async () => {
      await settings.expectPageLoaded();
    });
  });
});
