import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Collection Page Object
 *
 * Page: 3_Collection.py
 * Features:
 * - Object selector with progress bars
 * - Collection methods: ROS2 Camera, Local Camera, File Upload, Folder Import
 * - Captured images tree visualization
 * - Video frame extraction
 */
export class CollectionPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Collection';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Collection page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Object Selection
  // ============================================

  /**
   * Select an object to collect images for
   */
  async selectObject(objectName: string): Promise<void> {
    await this.selectOption('Select Object', objectName);
  }

  /**
   * Get currently selected object
   */
  async getSelectedObject(): Promise<string> {
    const selectbox = this.selectors.selectbox('Select Object');
    const selected = selectbox.locator('[data-baseweb="select"] span').first();
    return (await selected.textContent()) || '';
  }

  /**
   * Check if object selector is visible
   */
  async isObjectSelectorVisible(): Promise<boolean> {
    const selector = this.selectors.selectbox('Select Object');
    return await selector.isVisible().catch(() => false);
  }

  // ============================================
  // Collection Status
  // ============================================

  /**
   * Get collected images count
   */
  async getCollectedCount(): Promise<string> {
    return await this.getMetricValue('Collected');
  }

  /**
   * Get target count
   */
  async getTargetCount(): Promise<string> {
    return await this.getMetricValue('Target');
  }

  /**
   * Get progress percentage
   */
  async getProgress(): Promise<string> {
    return await this.getMetricValue('Progress');
  }

  // ============================================
  // Collection Method Tabs
  // ============================================

  /**
   * Click ROS2 Camera tab
   */
  async clickRos2CameraTab(): Promise<void> {
    await this.clickTabExact('🤖 ROS2 Camera');
  }

  /**
   * Click Local Camera tab
   */
  async clickLocalCameraTab(): Promise<void> {
    await this.clickTabExact('📷 Local Camera');
  }

  /**
   * Click File Upload tab
   */
  async clickFileUploadTab(): Promise<void> {
    await this.clickTabExact('📁 File Upload');
  }

  /**
   * Click Folder Import tab
   */
  async clickFolderImportTab(): Promise<void> {
    await this.clickTabExact('📂 Folder Import');
  }

  // ============================================
  // File Upload
  // ============================================

  /**
   * Upload image files
   */
  async uploadImages(filePaths: string | string[]): Promise<void> {
    await this.clickFileUploadTab();
    const uploader = this.selectors.fileUploader();
    const input = uploader.locator('input[type="file"]');
    await input.setInputFiles(filePaths);
    await this.wait.waitForFileUpload();
  }

  /**
   * Check if file uploader is visible
   */
  async isFileUploaderVisible(): Promise<boolean> {
    const uploader = this.selectors.fileUploader();
    return await uploader.isVisible().catch(() => false);
  }

  // ============================================
  // Folder Import
  // ============================================

  /**
   * Enter folder path for import
   */
  async enterFolderPath(path: string): Promise<void> {
    await this.clickFolderImportTab();
    await this.fillTextInput('Folder Path', path);
  }

  /**
   * Click import button
   */
  async clickImportButton(): Promise<void> {
    await this.clickButton('Import');
  }

  // ============================================
  // Reference Images
  // ============================================

  /**
   * Check if reference images section is visible
   */
  async isReferenceImagesSectionVisible(): Promise<boolean> {
    const section = this.page.getByText('Reference Images');
    return await section.isVisible().catch(() => false);
  }

  /**
   * Get reference images count
   */
  async getReferenceImagesCount(): Promise<number> {
    const images = this.mainContent.locator('[data-testid="stImage"]');
    return await images.count();
  }

  // ============================================
  // Captured Images Tree
  // ============================================

  /**
   * Check if captured images tree is visible
   */
  async isCapturedImagesTreeVisible(): Promise<boolean> {
    const tree = this.page.getByText('Captured Images');
    return await tree.isVisible().catch(() => false);
  }

  // ============================================
  // Video Extraction
  // ============================================

  /**
   * Check if video extraction section is visible
   */
  async isVideoExtractionVisible(): Promise<boolean> {
    const section = this.page.getByText(/Video.*Extract/i);
    return await section.isVisible().catch(() => false);
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert no objects warning is displayed
   */
  async expectNoObjectsWarning(): Promise<void> {
    await expect(this.page.getByText(/No objects registered/i)).toBeVisible();
  }

  /**
   * Assert collection status is displayed
   */
  async expectCollectionStatusDisplayed(): Promise<void> {
    await expect(this.selectors.metric('Collected')).toBeVisible();
    await expect(this.selectors.metric('Target')).toBeVisible();
  }

  /**
   * Assert collection tabs are visible
   */
  async expectCollectionTabsVisible(): Promise<void> {
    await expect(this.page.getByRole('tab', { name: /ROS2 Camera/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Local Camera/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /File Upload/i })).toBeVisible();
    await expect(this.page.getByRole('tab', { name: /Folder Import/i })).toBeVisible();
  }

  /**
   * Assert upload success message
   */
  async expectUploadSuccess(count?: number): Promise<void> {
    if (count !== undefined) {
      await expect(this.page.getByText(new RegExp(`Uploaded ${count} images`))).toBeVisible();
    } else {
      await expect(this.page.getByText(/Uploaded \d+ images/)).toBeVisible();
    }
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
  }
}
