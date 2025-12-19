import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './base.page';
import { SidebarComponent } from './sidebar.component';

/**
 * Registry Page Object
 *
 * Page: 2_Registry.py
 * Features:
 * - Object CRUD operations
 * - Thumbnail display
 * - Category filtering
 * - Edit/view mode toggle
 */
export class RegistryPage extends BasePage {
  readonly sidebar: SidebarComponent;
  readonly path = '/Registry';

  constructor(page: Page) {
    super(page);
    this.sidebar = new SidebarComponent(page);
  }

  /**
   * Navigate to Registry page
   */
  async goto(): Promise<void> {
    await this.navigate(this.path);
  }

  // ============================================
  // Tabs
  // ============================================

  /**
   * Click View Objects tab
   */
  async clickViewObjectsTab(): Promise<void> {
    await this.clickTab('View Objects');
  }

  /**
   * Click Add New Object tab
   */
  async clickAddNewObjectTab(): Promise<void> {
    await this.clickTab('Add New Object');
  }

  /**
   * Check which tab is active
   */
  async getActiveTab(): Promise<string> {
    const activeTab = this.page.locator('[data-baseweb="tab"][aria-selected="true"]');
    return (await activeTab.textContent()) || '';
  }

  // ============================================
  // View Objects Tab
  // ============================================

  /**
   * Filter objects by category
   */
  async filterByCategory(category: string): Promise<void> {
    await this.selectOption('Filter by Category', category);
  }

  /**
   * Get all visible object expanders
   */
  async getObjectExpanders(): Promise<Locator> {
    return this.mainContent.locator('[data-testid="stExpander"]');
  }

  /**
   * Get object count
   */
  async getObjectCount(): Promise<number> {
    const expanders = await this.getObjectExpanders();
    return await expanders.count();
  }

  /**
   * Expand an object by name
   */
  async expandObject(objectName: string): Promise<void> {
    await this.expandExpander(objectName);
  }

  /**
   * Click edit button for current object
   */
  async clickEditButton(): Promise<void> {
    await this.clickButton('Edit');
  }

  /**
   * Click save button after editing
   */
  async clickSaveButton(): Promise<void> {
    await this.clickButton('Save');
  }

  /**
   * Click cancel button
   */
  async clickCancelButton(): Promise<void> {
    await this.clickButton('Cancel');
  }

  /**
   * Click delete button
   */
  async clickDeleteButton(): Promise<void> {
    await this.clickButton('Delete');
  }

  // ============================================
  // Add New Object Tab
  // ============================================

  /**
   * Fill object registration form
   */
  async fillObjectForm(data: {
    name: string;
    displayName?: string;
    category: string;
    targetSamples?: number;
    isHeavy?: boolean;
    isTiny?: boolean;
    isLiquid?: boolean;
  }): Promise<void> {
    await this.fillTextInput('Object Name', data.name);

    if (data.displayName) {
      await this.fillTextInput('Display Name', data.displayName);
    }

    await this.selectOption('Category', data.category);

    if (data.targetSamples !== undefined) {
      await this.fillNumberInput('Target Samples', data.targetSamples);
    }

    if (data.isHeavy) {
      await this.toggleCheckbox('Heavy');
    }

    if (data.isTiny) {
      await this.toggleCheckbox('Tiny');
    }

    if (data.isLiquid) {
      await this.toggleCheckbox('Liquid');
    }
  }

  /**
   * Submit add object form
   */
  async submitAddObject(): Promise<void> {
    await this.clickButton('Add Object');
  }

  /**
   * Add a new object (complete flow)
   */
  async addObject(data: {
    name: string;
    displayName?: string;
    category: string;
    targetSamples?: number;
  }): Promise<void> {
    await this.clickAddNewObjectTab();
    await this.fillObjectForm(data);
    await this.submitAddObject();
  }

  // ============================================
  // Object Details
  // ============================================

  /**
   * Get object details when expanded
   */
  async getObjectDetails(objectName: string): Promise<{
    category?: string;
    collected?: string;
    target?: string;
  }> {
    await this.expandObject(objectName);

    const details: { category?: string; collected?: string; target?: string } = {};
    const expander = this.selectors.expander(objectName);

    // Try to extract details from the expanded content
    const categoryText = await expander.getByText(/Category:/i).textContent().catch(() => null);
    if (categoryText) {
      details.category = categoryText.replace(/Category:\s*/i, '');
    }

    return details;
  }

  // ============================================
  // Assertions
  // ============================================

  /**
   * Assert View Objects tab content is visible
   */
  async expectViewObjectsTabVisible(): Promise<void> {
    await expect(this.page.getByText('Filter by Category')).toBeVisible();
  }

  /**
   * Assert Add New Object tab content is visible
   */
  async expectAddNewObjectTabVisible(): Promise<void> {
    await expect(this.page.getByText('Object Name')).toBeVisible();
    await expect(this.page.getByRole('button', { name: 'Add Object' })).toBeVisible();
  }

  /**
   * Assert object exists in the list
   */
  async expectObjectExists(objectName: string): Promise<void> {
    await this.clickViewObjectsTab();
    const expander = this.selectors.expander(objectName);
    await expect(expander).toBeVisible();
  }

  /**
   * Assert object does not exist in the list
   */
  async expectObjectNotExists(objectName: string): Promise<void> {
    await this.clickViewObjectsTab();
    const expander = this.selectors.expander(objectName);
    await expect(expander).not.toBeVisible();
  }

  /**
   * Assert page is fully loaded
   */
  async expectPageLoaded(): Promise<void> {
    await this.wait.waitForAppLoad();
    await expect(this.appContainer).toBeVisible();
    await expect(this.selectors.tabs).toBeVisible();
  }
}
