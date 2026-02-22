import { Link } from 'react-router-dom';
import { FaGithub, FaBlog, FaEnvelope } from 'react-icons/fa';

const Footer = () => {
  return (
    <footer>
      <div className="mt-16 flex flex-col items-center">
        <div className="mb-3 flex space-x-4">
          <a href="mailto:sonjuhy@gmail.com" className="text-sm text-gray-500 transition hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
            <span className="sr-only">mail</span>
            <FaEnvelope className="h-6 w-6" />
          </a>
          <a href="https://github.com/sonjuhy" target="_blank" rel="noopener noreferrer" className="text-sm text-gray-500 transition hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
            <span className="sr-only">github</span>
            <FaGithub className="h-6 w-6" />
          </a>
          <a href="https://sonjuhy.tistory.com" target="_blank" rel="noopener noreferrer" className="text-sm text-gray-500 transition hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
            <span className="sr-only">blog</span>
            <FaBlog className="h-6 w-6" />
          </a>
        </div>
        <div className="mb-2 flex space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <div>Sonjuny</div>
          <div>{` • `}</div>
          <div>{`© ${new Date().getFullYear()}`}</div>
          <div>{` • `}</div>
          <Link to="/">AIStudy Blog</Link>
        </div>
        <div className="mb-8 text-sm text-gray-500 dark:text-gray-400">
          Sonjuhy AI Study Portfolio
        </div>
      </div>
    </footer>
  );
};

export default Footer;
