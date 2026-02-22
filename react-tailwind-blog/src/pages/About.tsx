import { FaGithub, FaTwitter, FaLinkedin, FaEnvelope } from 'react-icons/fa';

const About = () => {
  return (
    <>
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pb-8 pt-6 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            About
          </h1>
        </div>
        <div className="items-start space-y-2 xl:grid xl:grid-cols-3 xl:gap-x-8 xl:space-y-0 pt-8">
          <div className="flex flex-col items-center space-x-2 pt-8">
            <img src="https://ui-avatars.com/api/?name=Jane+Doe&size=192&background=random" alt="avatar" className="h-48 w-48 rounded-full" />
            <h3 className="pb-2 pt-4 text-2xl font-bold leading-8 tracking-tight">Jane Doe</h3>
            <div className="text-gray-500 dark:text-gray-400">Professor of Atmospheric Science</div>
            <div className="text-gray-500 dark:text-gray-400">Stanford University</div>
            <div className="flex space-x-3 pt-6">
              <a href="mailto:example@example.com" className="text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
                <span className="sr-only">mail</span>
                <FaEnvelope className="h-7 w-7" />
              </a>
              <a href="https://github.com" className="text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
                <span className="sr-only">github</span>
                <FaGithub className="h-7 w-7" />
              </a>
              <a href="https://linkedin.com" className="text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
                <span className="sr-only">linkedin</span>
                <FaLinkedin className="h-7 w-7" />
              </a>
              <a href="https://twitter.com" className="text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-300">
                <span className="sr-only">twitter</span>
                <FaTwitter className="h-7 w-7" />
              </a>
            </div>
          </div>
          <div className="prose max-w-none pb-8 pt-8 dark:prose-invert xl:col-span-2">
            <p>
              This is a clone of the Tailwind Nextjs Starter Blog created using React, Vite, and React Router. 
              The original template provides a robust starting point for technical blogs, offering out of the box features such as styling, MDX support, SEO optimization, and a lightweight footprint.
            </p>
            <p>
              In this React version, we emphasize the visual aesthetic and component structure to match the incredible look and feel of the Next.js original, while demonstrating how to achieve the necessary layouts in a standard SPA.
            </p>
            <p>
              Welcome to the TailwindBlog Starter project.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default About;
