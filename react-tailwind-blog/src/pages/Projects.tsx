import { useState } from 'react';
import { Link } from 'react-router-dom';

const MOCK_PROJECTS = [
  {
    title: 'CNN (Convolutional Neural Network)',
    description: 'Image classification and computer vision using Convolutional Neural Networks.',
    imgSrc: 'https://images.unsplash.com/photo-1542831371-29b0f74f9713?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
    href: '/projects/cnn',
  },
  {
    title: 'Custom YOLO',
    description: 'Real-time object detection implementation with custom trained YOLO (You Only Look Once) architecture.',
    imgSrc: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
    href: '/projects/custom-yolo',
  },
  {
    title: 'Emotion Detection',
    description: 'Facial expression recognition and sentiment analysis to identify human emotions accurately using deep learning.',
    imgSrc: 'https://images.unsplash.com/photo-1542831371-29b0f74f9713?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
    href: '/projects/emotion-detection',
  },
  {
    title: 'LLM (Large Language Model)',
    description: 'Integration and fine-tuning experiments with Large Language Models for advanced NLP tasks.',
    imgSrc: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
    href: '/projects/llm',
  },
  {
    title: 'RNN (Recurrent Neural Network)',
    description: 'Sequential data processing and time-series analysis utilizing Recurrent Neural Networks.',
    imgSrc: 'https://images.unsplash.com/photo-1542831371-29b0f74f9713?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80',
    href: '/projects/rnn',
  },
];

const Projects = () => {
  const [searchValue, setSearchValue] = useState('');

  const filteredProjects = MOCK_PROJECTS.filter((project) => {
    const searchContent = project.title + project.description;
    return searchContent.toLowerCase().includes(searchValue.toLowerCase());
  });

  return (
    <>
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pb-8 pt-6 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            Projects
          </h1>
          <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
            Showcase AI Study projects
          </p>
          <div className="relative max-w-lg pt-4">
            <input
              aria-label="Search projects"
              type="text"
              onChange={(e) => setSearchValue(e.target.value)}
              placeholder="Search projects"
              className="block w-full rounded-md border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-900 dark:bg-gray-800 dark:text-gray-100"
            />
            <svg
              className="absolute right-3 top-7 h-5 w-5 text-gray-400 dark:text-gray-300"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>
        <div className="container py-12">
          {!filteredProjects.length && (
            <div className="text-gray-500 dark:text-gray-400 pt-4">No projects found.</div>
          )}
          <div className="-m-4 flex flex-wrap">
            {filteredProjects.map((project) => (
              <div key={project.title} className="md max-w-[544px] p-4 md:w-1/2">
                <div className="flex h-full flex-col overflow-hidden rounded-md border-2 border-gray-200 border-opacity-60 dark:border-gray-700">
                  <Link to={project.href} aria-label={`Link to ${project.title}`}>
                    <img
                      alt={project.title}
                      src={project.imgSrc}
                      className="object-cover object-center md:h-36 lg:h-48"
                      width={544}
                      height={306}
                    />
                  </Link>
                  <div className="p-6">
                    <h2 className="mb-3 text-2xl font-bold leading-8 tracking-tight">
                      <Link to={project.href} aria-label={`Link to ${project.title}`}>
                        {project.title}
                      </Link>
                    </h2>
                    <p className="prose mb-3 max-w-none text-gray-500 dark:text-gray-400">
                      {project.description}
                    </p>
                    <Link
                      to={project.href}
                      className="text-base font-medium leading-6 text-primary-500 hover:text-primary-600 dark:hover:text-primary-400"
                      aria-label={`Link to ${project.title}`}
                    >
                      Learn more &rarr;
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
};

export default Projects;
