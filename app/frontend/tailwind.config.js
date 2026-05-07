/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        base: 'var(--bg-base)',
        surface: 'var(--bg-surface)',
        card: 'var(--bg-card)',
        hover: 'var(--bg-hover)',
        active: 'var(--bg-active)',
        border: {
          DEFAULT: 'var(--border)',
          md: 'var(--border-md)',
          hi: 'var(--border-hi)',
        },
        text: {
          primary: 'var(--text-primary)',
          sec: 'var(--text-sec)',
          muted: 'var(--text-muted)',
        },
        teal: {
          DEFAULT: 'var(--teal)',
          dim: 'var(--teal-dim)',
          mid: 'var(--teal-mid)',
        },
        green: {
          DEFAULT: 'var(--green)',
          dim: 'var(--green-dim)',
          mid: 'var(--green-mid)',
        },
        red: {
          DEFAULT: 'var(--red)',
          dim: 'var(--red-dim)',
          mid: 'var(--red-mid)',
        },
        amber: {
          DEFAULT: 'var(--amber)',
          dim: 'var(--amber-dim)',
          mid: 'var(--amber-mid)',
        },
        blue: {
          DEFAULT: 'var(--blue)',
          dim: 'var(--blue-dim)',
          mid: 'var(--blue-mid)',
        },
        violet: {
          DEFAULT: 'var(--violet)',
          dim: 'var(--violet-dim)',
        },
        orange: {
          DEFAULT: 'var(--orange)',
          dim: 'var(--orange-dim)',
          mid: 'var(--orange-mid)',
        },
      },
      fontFamily: {
        ui: 'var(--font-ui)',
        mono: 'var(--font-mono)',
        cond: 'var(--font-cond)',
      },
      borderRadius: {
        control: '4px',
        card: '6px',
        pill: '3px',
      },
      boxShadow: {
        card: 'var(--shadow-card)',
        elevated: 'var(--shadow-elevated)',
        overlay: 'var(--shadow-overlay)',
      },
      transitionDuration: {
        fast: 'var(--motion-fast)',
        base: 'var(--motion-base)',
        slow: 'var(--motion-slow)',
      },
    },
  },
  plugins: [],
}
