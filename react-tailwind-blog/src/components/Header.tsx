import { Link, useLocation } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';
import aiLogo from '../assets/ai.png';

const navLinks = [
  { href: '/blog', title: 'Blog' },
  { href: '/projects', title: 'Projects' },
];

const Header = () => {
  const location = useLocation();

  return (
    <header className="flex items-center justify-between py-10">
      <div>
        <Link to="/" aria-label="TailwindBlog">
          <div className="flex items-center justify-between">
            <div className="mr-3 flex items-center justify-center">
              <img src={aiLogo} alt="logo" className="h-10 w-10 object-cover rounded-full bg-white dark:bg-gray-800" />
            </div>
            <div className="hidden h-6 text-2xl font-semibold sm:block">
              AIStudy Blog
            </div>
          </div>
        </Link>
      </div>
      <div className="flex items-center space-x-4 leading-5 sm:space-x-6">
        {navLinks.map((link) => (
          <Link
            key={link.title}
            to={link.href}
            className={`hidden font-medium text-gray-900 dark:text-gray-100 sm:block ${
              location.pathname.startsWith(link.href)
                ? 'text-primary-500 dark:text-primary-400'
                : ''
            }`}
          >
            {link.title}
          </Link>
        ))}
        <ThemeToggle />
      </div>
    </header>
  );
};

export default Header;
