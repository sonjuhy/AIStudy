import { useEffect, useState } from 'react';
import { MdDarkMode, MdLightMode } from 'react-icons/md';

const ThemeToggle = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check local storage or system preference
    const isDark =
      localStorage.theme === 'dark' ||
      (!('theme' in localStorage) &&
        window.matchMedia('(prefers-color-scheme: dark)').matches);
    setIsDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleTheme = () => {
    const htmlClasses = document.documentElement.classList;
    if (isDarkMode) {
      htmlClasses.remove('dark');
      localStorage.theme = 'light';
      setIsDarkMode(false);
    } else {
      htmlClasses.add('dark');
      localStorage.theme = 'dark';
      setIsDarkMode(true);
    }
  };

  return (
    <button
      aria-label="Toggle Dark Mode"
      onClick={toggleTheme}
      className="p-1 sm:p-2 sm:ml-4 rounded-full bg-gray-200 dark:bg-gray-800 transition-colors"
    >
      {isDarkMode ? (
        <MdLightMode className="w-5 h-5 text-gray-900 dark:text-gray-100" />
      ) : (
        <MdDarkMode className="w-5 h-5 text-gray-900 dark:text-gray-100" />
      )}
    </button>
  );
};

export default ThemeToggle;
