/**
 * Playwright Global Setup for E2E Tests
 *
 * This script runs before all tests to set up the test environment:
 * 1. Backs up the existing profiles.json
 * 2. Creates a test profile directory (e2e_test)
 * 3. Copies fixture data to the test profile
 * 4. Updates profiles.json to use the test profile
 */

import { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.resolve(__dirname, '../../../../');
const FIXTURES_DIR = path.resolve(__dirname, '../');
const TARGET_PROFILES_DIR = path.join(PROJECT_ROOT, 'profiles');

/**
 * Required subdirectories for a valid profile
 * Based on ProfileManager.PROFILE_SUBDIRS
 */
const PROFILE_SUBDIRS = [
  'app_data',
  'app_data/thumbnails',
  'app_data/reference_images',
  'app_data/tasks',
  'datasets',
  'datasets/raw_captures',
  'datasets/annotated',
  'datasets/backgrounds',
  'datasets/videos',
  'models',
  'models/finetuned',
];

/**
 * Recursively copy a directory
 */
function copyRecursive(src: string, dest: string): void {
  if (!fs.existsSync(src)) return;

  fs.mkdirSync(dest, { recursive: true });

  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

async function globalSetup(config: FullConfig): Promise<void> {
  console.log('[E2E Setup] Starting test environment setup...');

  const profilesJsonPath = path.join(TARGET_PROFILES_DIR, 'profiles.json');
  const backupPath = path.join(TARGET_PROFILES_DIR, 'profiles.json.e2e-backup');
  const testProfileDir = path.join(TARGET_PROFILES_DIR, 'e2e_test');

  // 1. Backup existing profiles.json
  if (fs.existsSync(profilesJsonPath)) {
    console.log('[E2E Setup] Backing up existing profiles.json...');
    fs.copyFileSync(profilesJsonPath, backupPath);
  }

  // 2. Remove existing test profile if it exists
  if (fs.existsSync(testProfileDir)) {
    console.log('[E2E Setup] Removing existing test profile...');
    fs.rmSync(testProfileDir, { recursive: true });
  }

  // 3. Create test profile directory structure
  console.log('[E2E Setup] Creating test profile directory structure...');
  for (const subdir of PROFILE_SUBDIRS) {
    fs.mkdirSync(path.join(testProfileDir, subdir), { recursive: true });
  }

  // 4. Copy fixture files to test profile
  console.log('[E2E Setup] Copying fixture files...');
  const fixtureProfileDir = path.join(FIXTURES_DIR, 'profiles', 'test_profile');

  // Copy profile.json (rename id to e2e_test)
  const profileJsonSrc = path.join(fixtureProfileDir, 'profile.json');
  const profileJsonDest = path.join(testProfileDir, 'profile.json');
  fs.copyFileSync(profileJsonSrc, profileJsonDest);

  // Copy app_data files
  const appDataSrc = path.join(fixtureProfileDir, 'app_data');
  const appDataDest = path.join(testProfileDir, 'app_data');

  for (const file of ['object_registry.json', 'ui_settings.json']) {
    const srcFile = path.join(appDataSrc, file);
    const destFile = path.join(appDataDest, file);
    if (fs.existsSync(srcFile)) {
      fs.copyFileSync(srcFile, destFile);
    }
  }

  // 5. Create test profiles.json
  console.log('[E2E Setup] Creating test profiles.json...');
  const testProfilesJson = {
    version: '1.0.0',
    active_profile_id: 'e2e_test',
    profiles: [
      {
        id: 'e2e_test',
        display_name: 'E2E Test Profile',
        created_at: new Date().toISOString(),
        last_accessed: null,
        description: 'Playwright E2E test profile',
      },
    ],
  };

  fs.writeFileSync(profilesJsonPath, JSON.stringify(testProfilesJson, null, 2));

  // 6. Set environment variables
  process.env.HSR_TEST_MODE = 'true';
  process.env.HSR_TEST_PROFILE = 'e2e_test';

  console.log('[E2E Setup] Test environment setup complete.');
  console.log(`[E2E Setup] Test profile created at: ${testProfileDir}`);
}

export default globalSetup;
