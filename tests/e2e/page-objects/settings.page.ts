import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Settings Page Object
 *
 * Page: 7_Settings.py
 * Features:
 * - Profile management (create, switch, export, import)
 * - Data management (export YOLO config, update counts)
 * - Category management
 * - System status display
 * - Data paths display
 */
export class SettingsPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Settings';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Settings page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Profile Management Tabs
  // ============================================

  /**
   * Click Profile List tab
   */
  async clickProfileListTab(): Promise<void> {
    await this.clickTabExact('Profile List');
  }

  /**
   * Click Create New tab
   */
  async clickCreateNewTab(): Promise<void> {
    await this.clickTabExact('Create New');
  }

  /**
   * Click Import/Export tab
   */
  async clickImportExportTab(): Promise<void> {
    await this.clickTabExact('Import/Export');
  }

  // Aliases for backward compatibility
  async clickSwitchTab(): Promise<void> {
    await this.clickProfileListTab();
  }

  async clickCreateTab(): Promise<void> {
    await this.clickCreateNewTab();
  }

  async clickExportTab(): Promise<void> {
    await this.clickImportExportTab();
  }

  async clickImportTab(): Promise<void> {
    await this.clickImportExportTab();
  }

  // ============================================
  // Profile Operations
  // ============================================

  /**
   * Create a new profile
   */
  async createProfile(name: string, description?: string): Promise<void> {
    await this.clickCreateTab();
    await this.fillTextInput('Profile Name', name);

    if (description) {
      await this.fillTextInput('Description', description);
    }

    await this.clickButton('Create Profile');
  }

  /**
   * Switch to a different profile
   */
  async switchProfile(profileName: string): Promise<void> {
    await this.clickSwitchTab();
    await this.selectOption('Select Profile', profileName);
    await this.clickButton('Switch Profile');
  }

  /**
   * Export current profile
   */
  async exportProfile(): Promise<void> {
    await this.clickExportTab();
    await this.clickButton('Export Profile');
  }

  /**
   * Import a profile from file
   */
  async importProfile(filePath: string): Promise<void> {
    await this.clickImportTab();
    await this.uploadFile(filePath);
    await this.clickButton('Import');
  }

  /**
   * Get current profile name
   */
  async getCurrentProfileName(): Promise<string> {
    // Look for current profile indicator
    const profileInfo = this.page.getByText(/Current Profile:/i);
    if (await profileInfo.isVisible().catch(() => false)) {
      const text = await profileInfo.textContent();
      return text?.replace(/Current Profile:\s*/i, '') || '';
    }
    return '';
  }

  // ============================================
  // Data Management
  // ============================================

  /**
   * Export to YOLO config
   */
  async exportToYoloConfig(): Promise<void> {
    await this.clickButton('Export to YOLO Config');
  }

  /**
   * Update all collection counts
   */
  async updateAllCollectionCounts(): Promise<void> {
    await this.clickButton('Update All Collection Counts');
  }

  // ============================================
  // Category Management
  // ============================================

  /**
   * Add a new category
   */
  async addCategory(categoryName: string): Promise<void> {
    await this.fillTextInput('New Category', categoryName);
    await this.clickButton('Add Category');
  }

  /**
   * Get list of existing categories
   */
  async getCategories(): Promise<string[]> {
    const categoryList = this.page.locator('text=/Categories:/i').locator('..');
    const text = await categoryList.textContent();
    // Parse categories from text
    const match = text?.match(/Categories:\s*(.+)/i);
    if (match) {
      return match[1].split(',').map((c) => c.trim());
    }
    return [];
  }

  // ============================================
  // System Status
  // ============================================

  /**
   * Get system status section
   */
  async getSystemStatusSection(): Promise<Locator> {
    return this.page.getByText('System Status').locator('..');
  }

  /**
   * Check if system status is visible
   */
  async isSystemStatusVisible(): Promise<boolean> {
    const section = this.page.getByText('System Status');
    return await section.isVisible().catch(() => false);
  }

  /**
   * Get GPU availability status
   */
  async isGpuAvailable(): Promise<boolean> {
    const gpuStatus = this.page.getByText(/GPU.*Available|CUDA/i);
    return await gpuStatus.isVisible().catch(() => false);
  }

  /**
   * Get ROS2 availability status
   */
  async isRos2Available(): Promise<boolean> {
    const ros2Status = this.page.getByText(/ROS2.*Available/i);
    return await ros2Status.isVisible().catch(() => false);
  }

  // ============================================
  // Data Paths
  // ============================================

  /**
   * Get data paths section
   */
  async getDataPathsSection(): Promise<Locator> {
    return this.page.getByText('Data Paths').locator('..');
  }

  /**
   * Check if data paths are visible
   */
  async isDataPathsVisible(): Promise<boolean> {
    const section = this.page.getByText('Data Paths');
    return await section.isVisible().catch(() => false);
  }

  /**
   * Get specific path value
   */
  async getPathValue(pathLabel: string): Promise<string> {
    const pathRow = this.page.getByText(pathLabel).locator('..');
    return (await pathRow.textContent()) || '';
  }

  // ============================================
  // Danger Zone
  // ============================================

  /**
   * Delete current profile (if supported)
   */
  async deleteCurrentProfile(): Promise<void> {
    // This might require confirmation
    await this.clickButton('Delete Profile');
  }

  /**
   * Reset all settings
   */
  async resetSettings(): Promise<void> {
    await this.clickButton('Reset Settings');
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert profile management section is visible
   */
  async expectProfileManagementVisible(): Promise<void> {
    // Use heading role to be more specific about "Profile Management" section
    await expect(this.page.getByRole('heading', { name: /Profile Management/i })).toBeVisible();
  }

  /**
   * Assert profile creation tabs are visible
   */
  async expectProfileTabsVisible(): Promise<void> {
    await expect(this.page.getByRole('tab', { name: /Profile List/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Create New/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Import\/Export/i })).toBeVisible();
  }

  /**
   * Assert data management options are visible
   */
  async expectDataManagementVisible(): Promise<void> {
    await expect(this.page.getByRole('button', { name: /Export to YOLO/i })).toBeVisible();
    await expect(this.page.getByRole('button', { name: /Update.*Collection/i })).toBeVisible();
  }

  /**
   * Assert profile created successfully
   */
  async expectProfileCreated(): Promise<void> {
    await this.expectSuccessAlert(/created|success/i);
  }

  /**
   * Assert category added successfully
   */
  async expectCategoryAdded(): Promise<void> {
    await this.expectSuccessAlert(/Added category/i);
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await this.expectProfileManagementVisible();
  }
}
