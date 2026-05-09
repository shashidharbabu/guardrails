import { expect, test } from '@playwright/test'

const sessionUser = {
  email: 'qa@spartanguard.local',
  name: 'QA Operator',
  org: 'SpartanGuard Workspace',
  role: 'Operator',
}

async function signIn(page) {
  await page.goto('/sign-in')
  await page.getByLabel('Work email').fill(sessionUser.email)
  await page.getByLabel('Password').fill('local-dev-password')
  await page.getByRole('button', { name: 'Enter console' }).click()
  await expect(page).toHaveURL(/\/conversations$/)
  await expect(page.getByText('SpartanGuard').first()).toBeVisible()
}

async function seedSession(page) {
  const response = await page.request.post('/api/query', {
    data: {
      query: 'What are HIPAA data retention requirements for clinical records?',
      llm_model: 'qwen2.5:7b',
    },
    timeout: 45_000,
  })
  expect(response.ok()).toBeTruthy()
  const session = await response.json()
  expect(session.id).toBeTruthy()
  return session
}

test.describe('SpartanGuard full product flow', () => {
  test.beforeEach(async ({ page }) => {
    const consoleErrors = []
    page.on('console', msg => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', err => {
      consoleErrors.push(err.message)
    })
    page.consoleErrors = consoleErrors
  })

  test('public landing, auth CTAs, and app shell load', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/SpartanGuard/)
    await expect(page.getByText('SpartanGuard').first()).toBeVisible()
    await expect(page.locator('.brand-mark').first()).toHaveCSS('background-image', /spartanguard-logo\.svg/)
    await page.mouse.wheel(0, 900)
    await expect.poll(() => page.locator('.public-page').evaluate(el => el.scrollTop)).toBeGreaterThan(0)

    await page.getByLabel('Public navigation').getByRole('link', { name: 'Request access' }).click()
    await expect(page).toHaveURL(/\/request-access$/)
    await expect(page.getByRole('heading', { name: 'Request access to SpartanGuard.' })).toBeVisible()

    await page.getByRole('link', { name: 'Sign in' }).click()
    await expect(page).toHaveURL(/\/sign-in$/)
    await expect(page.getByRole('heading', { name: 'Sign in to SpartanGuard.' })).toBeVisible()

    await signIn(page)
    await expect(page.getByRole('navigation', { name: 'Main navigation' })).toBeVisible()
    await expect(page.locator('main#main-content')).toBeVisible()
    expect(page.consoleErrors).toEqual([])
  })

  test('primary navigation routes load and key controls are interactive', async ({ page }) => {
    await signIn(page)

    const routes = [
      ['Conversations', /\/conversations$/, 'Conversations'],
      ['Gateway', /\/gateway$/, 'Gateway'],
      ['Analytics', /\/analytics$/, 'Analytics'],
      ['Human Review', /\/human-review$/, 'Human Review Queue'],
      ['Feedback', /\/feedback$/, 'Review Queue'],
      ['Evaluation', /\/evaluation$/, 'Evaluation'],
      ['Test Query', /\/new$/, 'Test Query'],
      ['System Health', /\/system-health$/, 'System Health'],
      ['Audit Logs', /\/audit$/, 'Audit Logs'],
      ['Settings', /\/settings$/, 'Settings'],
    ]

    for (const [navLabel, url, heading] of routes) {
      await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: new RegExp(navLabel) }).click()
      await expect(page).toHaveURL(url)
      await expect(page.getByText(heading, { exact: true }).first()).toBeVisible()
    }

    await page.goto('/conversations')
    await page.getByPlaceholder('Search sessions…').fill('HIPAA')
    await page.getByRole('button', { name: 'BLOCK' }).click()
    await page.getByRole('button', { name: 'Clear filters' }).click()
    await expect(page.getByPlaceholder('Search sessions…')).toHaveValue('')
    await page.getByLabel('Search sessions, policies, traces').fill('HIPAA data retention')
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/\/conversations\?q=HIPAA\+data\+retention|\/conversations\?q=HIPAA%20data%20retention/)
    await expect(page.getByPlaceholder('Search sessions…')).toHaveValue('HIPAA data retention')

    await page.goto('/gateway')
    await page.getByRole('button', { name: 'Audit Logs' }).click()
    await expect(page.getByRole('button', { name: 'Refresh' })).toBeVisible()
    await page.getByRole('button', { name: 'Stats' }).click()
    await expect(page.getByText('Decision Distribution')).toBeVisible()
    await page.getByRole('button', { name: 'Model Config' }).click()
    await expect(page.getByText('Active Models')).toBeVisible()
    await page.getByRole('button', { name: 'Live Test' }).click()
    await expect(page.getByText('Enter text or select a preset scenario')).toBeVisible()

    await page.goto('/system-health')
    await page.getByRole('button', { name: /Refresh/ }).click()
    await expect(page.getByText('database').first()).toBeVisible()

    expect(page.consoleErrors).toEqual([])
  })

  test('gateway live validation degrades safely when validators are unavailable', async ({ page }) => {
    await signIn(page)
    await page.goto('/gateway')
    await page.locator('select.q-select').first().selectOption({ label: 'Classic DAN jailbreak' })
    await page.getByRole('button', { name: /Run Gateway/ }).click()
    await expect(page.getByText('VALIDATOR_ERROR', { exact: true })).toBeVisible({ timeout: 25_000 })
    await expect(page.getByText(/Flagged for analyst review/).first()).toBeVisible()
    expect(page.consoleErrors).toEqual([])
  })

  test('query submission persists and opens a session trace', async ({ page }) => {
    await signIn(page)
    await page.goto('/new')
    await page.getByRole('button', { name: /HIPAA retention/ }).click()
    await page.getByRole('button', { name: /Run Pipeline/ }).click()
    await expect(page.getByText('Result')).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: /View full trace/ }).click()
    await expect(page).toHaveURL(/\/sessions\/[a-f0-9-]+$/)
    await expect(page.getByRole('link', { name: /Back to conversations/ })).toBeVisible()
    await expect(page.getByText('Gateway Layer')).toBeVisible()
    await expect(page.getByText(/Multi-Agent Debate unavailable|LLM candidate answer/).first()).toBeVisible()
    const main = page.locator('main#main-content')
    await main.evaluate(el => { el.scrollTop = 0 })
    await main.hover()
    await page.mouse.wheel(0, 900)
    await expect.poll(() => main.evaluate(el => el.scrollTop)).toBeGreaterThan(0)

    await page.goto('/conversations')
    await page.getByPlaceholder('Search sessions…').fill('HIPAA data retention')
    await expect(page.getByText('What are HIPAA data retention requirements').first()).toBeVisible()
    expect(page.consoleErrors).toEqual([])
  })

  test('seeded session detail supports feedback and audit endpoints through UI', async ({ page }) => {
    await signIn(page)
    const session = await seedSession(page)
    await page.goto(`/sessions/${session.id}`)
    await expect(page.getByRole('link', { name: /Back to conversations/ })).toBeVisible()
    await page.getByRole('button', { name: 'Rate 4' }).click()
    await page.getByRole('button', { name: 'correct', exact: true }).click()
    await page.getByPlaceholder('Optional note…').fill('Browser E2E feedback check.')
    await page.getByRole('button', { name: /Submit Feedback/ }).click()
    await expect(page.getByText('Feedback submitted. Thank you.')).toBeVisible()

    await page.goto('/audit')
    await expect(page.getByText('Audit Logs', { exact: true }).first()).toBeVisible()
    await expect(page.getByText(/query_submitted|feedback_created/).first()).toBeVisible()
    expect(page.consoleErrors).toEqual([])
  })
})
