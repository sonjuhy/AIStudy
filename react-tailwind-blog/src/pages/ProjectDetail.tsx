import { useParams, Link } from 'react-router-dom';

const MOCK_PROJECTS_DATA: Record<string, { date: string, title: string, tags: string[], author: string }> = {
  'cnn': { date: '2023-10-15', title: 'CNN (Convolutional Neural Network)', tags: ['deep-learning', 'computer-vision', 'image-classification'], author: 'AI Researcher' },
  'custom-yolo': { date: '2023-11-02', title: 'Custom YOLO', tags: ['object-detection', 'yolo', 'real-time'], author: 'Vision Engineer' },
  'emotion-detection': { date: '2023-12-20', title: 'Emotion Detection', tags: ['facial-expression', 'sentiment-analysis', 'ai'], author: 'AI Developer' },
  'llm': { date: '2024-01-10', title: 'LLM (Large Language Model)', tags: ['nlp', 'transformers', 'generative-ai'], author: 'NLP Specialist' },
  'rnn': { date: '2024-02-05', title: 'RNN (Recurrent Neural Network)', tags: ['time-series', 'sequence-modeling', 'deep-learning'], author: 'Data Scientist' },
};

const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const projectSlug = id || '';
  const project = MOCK_PROJECTS_DATA[projectSlug] || {
    date: '2024-01-01',
    title: projectSlug.replace(/-/g, ' ').toUpperCase() || 'Project Title',
    tags: ['project', 'ai'],
    author: 'Author Name'
  };

  return (
    <article>
      <div className="xl:divide-y xl:divide-gray-200 xl:dark:divide-gray-700">
        <header className="pt-6 xl:pb-6">
          <div className="space-y-1 text-center">
            <dl className="space-y-10">
              <div>
                <dt className="sr-only">Published on</dt>
                <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                  <time dateTime={project.date}>
                    {new Date(project.date).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </time>
                </dd>
              </div>
            </dl>
            <div>
              <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-5xl md:leading-14">
                {project.title}
              </h1>
            </div>
          </div>
        </header>
        <div className="grid-rows-[auto_1fr] divide-y divide-gray-200 pb-8 dark:divide-gray-700 xl:grid xl:grid-cols-4 xl:gap-x-6 xl:divide-y-0">
          <dl className="pb-10 pt-6 xl:border-b xl:border-gray-200 xl:pt-11 xl:dark:border-gray-700">
            <dt className="sr-only">Authors</dt>
            <dd>
              <ul className="flex flex-wrap justify-center gap-4 sm:space-x-12 xl:block xl:space-x-0 xl:space-y-8">
                <li className="flex items-center space-x-2">
                  <img src={`https://ui-avatars.com/api/?name=${project.author}&background=random`} alt="avatar" className="h-10 w-10 rounded-full" />
                  <dl className="whitespace-nowrap text-sm font-medium leading-5">
                    <dt className="sr-only">Name</dt>
                    <dd className="text-gray-900 dark:text-gray-100">{project.author}</dd>
                    <dt className="sr-only">Twitter</dt>
                    <dd>
                      <a href="#" className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">@twitter</a>
                    </dd>
                  </dl>
                </li>
              </ul>
            </dd>
          </dl>
          <div className="divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-3 xl:row-span-2 xl:pb-0">
            <div className="prose max-w-none pb-8 pt-10 dark:prose-invert">
              <p>This is a detailed page for the <strong>{project.title}</strong> project.</p>
              <p>Just like the Next.js starter blog layout, this space is reserved for deeply explaining the project goals, architecture, methodologies used, and the eventual outcomes.</p>
              <h2>Background</h2>
              <p>We started exploring this architecture because it offers unparalleled accuracy for this specific subset of problems in the AI field...</p>
              <h2>Implementation Detail</h2>
              <pre><code>{`// Pseudo-code or snippet showcase\ndef train_model(data):\n    model = init_structure()\n    model.fit(data.x, data.y, epochs=100)\n    return model`}</code></pre>
              <p>The code syntax highlighting could be handled via additional libraries like <code>prismjs</code> or <code>highlight.js</code>.</p>
            </div>
          </div>
          <footer className="divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-1 xl:row-start-2">
            <div className="py-4 xl:py-8">
              <h2 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Tags</h2>
              <div className="flex flex-wrap">
                {project.tags.map(tag => (
                  <Link key={tag} to={`/tags/${tag}`} className="mr-3 text-sm font-medium uppercase text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">
                    {tag}
                  </Link>
                ))}
              </div>
            </div>
            <div className="pt-4 xl:pt-8">
              <Link to="/" className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">
                &larr; Back to projects
              </Link>
            </div>
          </footer>
        </div>
      </div>
    </article>
  );
};

export default ProjectDetail;
