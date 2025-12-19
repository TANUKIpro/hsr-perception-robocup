import { test, expect } from '@playwright/test';
import { BasePage } from '../../page-objects/base.page';
import { SidebarComponent } from '../../page-objects/sidebar.component';

/**
 * App Launch Smoke Tests
 *
 * Basic tests to verify the Streamlit application launches correctly
 * and core components are visible.
 */
test.describe('App Launch', () => {
  let basePage: BasePage;
  let sidebar: SidebarComponent;

  test.beforeEach(async ({ page }) => {
    basePage = new BasePage(page);
    sidebar = new SidebarComponent(page);
  });

  test('should load the home page successfully', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Verify app container is visible
    await expect(basePage.appContainer).toBeVisible();
  });

  test('should display the application title', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Look for HSR Object Manager title (can be in h1 or markdown with HTML)
    // The title is rendered via st.markdown with unsafe_allow_html=True
    const title = page.locator('h1:has-text("HSR Object Manager"), [data-testid="stMarkdown"]:has-text("HSR Object Manager")').first();
    await expect(title).toBeVisible();
  });

  test('should display the sidebar', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Verify sidebar is visible
    await expect(sidebar.container).toBeVisible();
  });

  test('should display navigation links in sidebar', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Check for main navigation links
    const navLinks = await sidebar.getNavigationLinks();
    expect(navLinks.length).toBeGreaterThan(0);

    // Should have Dashboard link at minimum
    const hasDashboard = navLinks.some((link) => link.includes('Dashboard'));
    expect(hasDashboard).toBe(true);
  });

  test('should display profile selector', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Check if profile selector is visible
    const isProfileSelectorVisible = await sidebar.isProfileSelectorVisible();
    expect(isProfileSelectorVisible).toBe(true);
  });

  test('should have no console errors on load', async ({ page }) => {
    const consoleErrors: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Filter out known acceptable errors (e.g., favicon)
    const criticalErrors = consoleErrors.filter(
      (error) => !error.includes('favicon') && !error.includes('404')
    );

    expect(criticalErrors).toHaveLength(0);
  });

  test('should respond within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    const loadTime = Date.now() - startTime;

    // App should load within 30 seconds
    expect(loadTime).toBeLessThan(30000);
  });

  test('should have correct viewport', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    const viewport = page.viewportSize();
    // Verify viewport is set and has reasonable dimensions for desktop
    // Note: Playwright may use different viewport sizes depending on configuration
    expect(viewport).toBeDefined();
    expect(viewport?.width).toBeGreaterThanOrEqual(1024); // Minimum desktop width
    expect(viewport?.height).toBeGreaterThanOrEqual(600); // Minimum desktop height
  });

  test('should handle page refresh gracefully', async ({ page }) => {
    await basePage.navigate('/');
    await basePage.wait.waitForAppLoad();

    // Refresh the page
    await page.reload();
    await basePage.wait.waitForAppLoad();

    // App should still be functional
    await expect(basePage.appContainer).toBeVisible();
    await expect(sidebar.container).toBeVisible();
  });
});
