/**
 * Playwright Global Teardown for E2E Tests
 *
 * This script runs after all tests to clean up the test environment:
 * 1. Removes the test profile directory (e2e_test)
 * 2. Restores the original profiles.json from backup
 * 3. Cleans up environment variables
 */

import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.resolve(__dirname, '../../../../');
const TARGET_PROFILES_DIR = path.join(PROJECT_ROOT, 'profiles');

async function globalTeardown(config: FullConfig): Promise<void> {
  console.log('[E2E Teardown] Starting test environment cleanup...');

  const profilesJsonPath = path.join(TARGET_PROFILES_DIR, 'profiles.json');
  const backupPath = path.join(TARGET_PROFILES_DIR, 'profiles.json.e2e-backup');
  const testProfileDir = path.join(TARGET_PROFILES_DIR, 'e2e_test');

  // 1. Remove test profile directory
  if (fs.existsSync(testProfileDir)) {
    console.log('[E2E Teardown] Removing test profile directory...');
    fs.rmSync(testProfileDir, { recursive: true });
  }

  // 2. Restore original profiles.json from backup
  if (fs.existsSync(backupPath)) {
    console.log('[E2E Teardown] Restoring original profiles.json...');
    fs.copyFileSync(backupPath, profilesJsonPath);
    fs.unlinkSync(backupPath);
  } else {
    // No backup exists - this is a fresh environment
    // Remove the test profiles.json if it was created
    console.log('[E2E Teardown] No backup found, removing test profiles.json...');
    if (fs.existsSync(profilesJsonPath)) {
      // Check if it's our test profiles.json
      try {
        const content = JSON.parse(fs.readFileSync(profilesJsonPath, 'utf-8'));
        if (content.active_profile_id === 'e2e_test') {
          fs.unlinkSync(profilesJsonPath);
        }
      } catch (e) {
        // If we can't read it, leave it alone
        console.log('[E2E Teardown] Could not verify profiles.json, leaving it as is.');
      }
    }
  }

  // 3. Clean up environment variables
  delete process.env.HSR_TEST_MODE;
  delete process.env.HSR_TEST_PROFILE;

  console.log('[E2E Teardown] Test environment cleanup complete.');
}

export default globalTeardown;
