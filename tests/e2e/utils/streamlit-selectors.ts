import { Locator, Page } from '@playwright/test';

/**
 * Streamlit Component Selectors
 *
 * Streamlit generates specific data-testid attributes for components.
 * This utility provides stable selectors for common Streamlit components.
 *
 * Reference: Streamlit uses data-testid for component identification
 */
export class StreamlitSelectors {
  constructor(private page: Page) {}

  // ============================================
  // Layout Elements
  // ============================================

  /** Main app container */
  get appContainer(): Locator {
    return this.page.locator('[data-testid="stAppViewContainer"]');
  }

  /** Sidebar container */
  get sidebar(): Locator {
    return this.page.locator('[data-testid="stSidebar"]');
  }

  /** Main content block */
  get mainContent(): Locator {
    return this.page.locator('[data-testid="stMainBlockContainer"]');
  }

  /** Loading spinner */
  get spinner(): Locator {
    return this.page.locator('[data-testid="stSpinner"]');
  }

  /** Header element */
  get header(): Locator {
    return this.page.locator('[data-testid="stHeader"]');
  }

  // ============================================
  // Form Elements
  // ============================================

  /**
   * Text input by label
   */
  textInput(label: string): Locator {
    return this.page.locator(`[data-testid="stTextInput"]:has-text("${label}") input`);
  }

  /**
   * Number input by label
   */
  numberInput(label: string): Locator {
    return this.page.locator(`[data-testid="stNumberInput"]:has-text("${label}") input`);
  }

  /**
   * Selectbox by label
   */
  selectbox(label: string): Locator {
    return this.page.locator(`[data-testid="stSelectbox"]:has-text("${label}")`);
  }

  /**
   * Multiselect by label
   */
  multiselect(label: string): Locator {
    return this.page.locator(`[data-testid="stMultiSelect"]:has-text("${label}")`);
  }

  /**
   * Checkbox by label
   */
  checkbox(label: string): Locator {
    return this.page.locator(`[data-testid="stCheckbox"]:has-text("${label}")`);
  }

  /**
   * Slider by label
   */
  slider(label: string): Locator {
    return this.page.locator(`[data-testid="stSlider"]:has-text("${label}")`);
  }

  /**
   * Radio buttons by label
   */
  radio(label: string): Locator {
    return this.page.locator(`[data-testid="stRadio"]:has-text("${label}")`);
  }

  /**
   * File uploader by label (optional)
   */
  fileUploader(label?: string): Locator {
    if (label) {
      return this.page.locator(`[data-testid="stFileUploader"]:has-text("${label}")`);
    }
    return this.page.locator('[data-testid="stFileUploader"]');
  }

  /**
   * Button by text
   */
  button(text: string): Locator {
    return this.page.getByRole('button', { name: text });
  }

  /**
   * Form submit button by text
   */
  submitButton(text: string): Locator {
    return this.page.locator(`[data-testid="stFormSubmitButton"] button:has-text("${text}")`);
  }

  // ============================================
  // Display Elements
  // ============================================

  /**
   * Tabs container
   */
  get tabs(): Locator {
    return this.page.locator('[data-testid="stTabs"]');
  }

  /**
   * Tab by text
   */
  tab(text: string): Locator {
    return this.page.locator(`[data-baseweb="tab"]:has-text("${text}")`);
  }

  /**
   * Expander by title
   */
  expander(title: string): Locator {
    return this.page.locator(`[data-testid="stExpander"]:has-text("${title}")`);
  }

  /**
   * Metric by label
   */
  metric(label: string): Locator {
    return this.page.locator(`[data-testid="stMetric"]:has-text("${label}")`);
  }

  /**
   * Metric value element
   */
  metricValue(label: string): Locator {
    return this.metric(label).locator('[data-testid="stMetricValue"]');
  }

  /**
   * Progress bar
   */
  get progressBar(): Locator {
    return this.page.locator('[data-testid="stProgress"]');
  }

  /**
   * Image element
   */
  get image(): Locator {
    return this.page.locator('[data-testid="stImage"]');
  }

  // ============================================
  // Feedback Elements
  // ============================================

  /**
   * Success alert
   */
  get alertSuccess(): Locator {
    return this.page.locator('[data-testid="stAlert"][data-baseweb="notification"][kind="positive"]');
  }

  /**
   * Error alert
   */
  get alertError(): Locator {
    return this.page.locator('[data-testid="stAlert"][data-baseweb="notification"][kind="negative"]');
  }

  /**
   * Warning alert
   */
  get alertWarning(): Locator {
    return this.page.locator('[data-testid="stAlert"][data-baseweb="notification"][kind="warning"]');
  }

  /**
   * Info alert
   */
  get alertInfo(): Locator {
    return this.page.locator('[data-testid="stAlert"][data-baseweb="notification"][kind="info"]');
  }

  /**
   * Toast notification
   */
  get toast(): Locator {
    return this.page.locator('[data-testid="stToast"]');
  }

  // ============================================
  // Navigation Elements
  // ============================================

  /**
   * Page link in sidebar by page name
   */
  pageLink(pageName: string): Locator {
    return this.sidebar.locator(`a[href*="${pageName}"], [data-testid="stSidebarNavLink"]:has-text("${pageName}")`);
  }

  /**
   * All sidebar nav links
   */
  get navLinks(): Locator {
    return this.sidebar.locator('[data-testid="stSidebarNavLink"]');
  }

  // ============================================
  // Special Components
  // ============================================

  /**
   * Download button by text
   */
  downloadButton(text: string): Locator {
    return this.page.locator(`[data-testid="stDownloadButton"]:has-text("${text}")`);
  }

  /**
   * Camera input (hidden file input)
   */
  get cameraInput(): Locator {
    return this.page.locator('[data-testid="stCameraInput"] input[type="file"]');
  }

  /**
   * Column by index (0-based)
   */
  column(index: number): Locator {
    return this.page.locator(`[data-testid="stColumn"]:nth-child(${index + 1})`);
  }

  /**
   * Container with border
   */
  get container(): Locator {
    return this.page.locator('[data-testid="stVerticalBlock"]');
  }
}

/**
 * Selector constants for direct use
 */
export const STREAMLIT_SELECTORS = {
  APP_CONTAINER: '[data-testid="stAppViewContainer"]',
  SIDEBAR: '[data-testid="stSidebar"]',
  MAIN_CONTENT: '[data-testid="stMainBlockContainer"]',
  SPINNER: '[data-testid="stSpinner"]',
  HEADER: '[data-testid="stHeader"]',
  TABS: '[data-testid="stTabs"]',
  EXPANDER: '[data-testid="stExpander"]',
  METRIC: '[data-testid="stMetric"]',
  PROGRESS: '[data-testid="stProgress"]',
  TOAST: '[data-testid="stToast"]',
  FILE_UPLOADER: '[data-testid="stFileUploader"]',
  SELECTBOX: '[data-testid="stSelectbox"]',
  TEXT_INPUT: '[data-testid="stTextInput"]',
  NUMBER_INPUT: '[data-testid="stNumberInput"]',
  CHECKBOX: '[data-testid="stCheckbox"]',
  SLIDER: '[data-testid="stSlider"]',
  RADIO: '[data-testid="stRadio"]',
} as const;
