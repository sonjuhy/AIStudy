import { useParams, Link } from 'react-router-dom';

const BlogDetail = () => {
  const { id } = useParams<{ id: string }>();

  return (
    <article>
      <div className="xl:divide-y xl:divide-gray-200 xl:dark:divide-gray-700">
        <header className="pt-6 xl:pb-6">
          <div className="space-y-1 text-center">
            <dl className="space-y-10">
              <div>
                <dt className="sr-only">Published on</dt>
                <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                  <time dateTime="2023-08-05">August 5, 2023</time>
                </dd>
              </div>
            </dl>
            <div>
              <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-5xl md:leading-14">
                {id?.replace(/-/g, ' ').toUpperCase() || 'Blog Post Title'}
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
                  <img src="https://ui-avatars.com/api/?name=Author&background=random" alt="avatar" className="h-10 w-10 rounded-full" />
                  <dl className="whitespace-nowrap text-sm font-medium leading-5">
                    <dt className="sr-only">Name</dt>
                    <dd className="text-gray-900 dark:text-gray-100">Temp Author</dd>
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
              <p>This is a placeholder for the blog content. The post slug is <strong>{id}</strong>.</p>
              <p>In a real application, you would fetch the mdx content or markdown files and render them here, utilizing components like <code>react-markdown</code> or <code>mdx-bundler</code>.</p>
              <h2>Heading 2</h2>
              <p>Tailwind Nextjs Starter Blog heavily utilizes Tailwind Typography for the prose styling. It makes formatting article content incredibly easy and visually appealing.</p>
            </div>
          </div>
          <footer className="divide-y divide-gray-200 dark:divide-gray-700 xl:col-span-1 xl:row-start-2">
            <div className="py-4 xl:py-8">
              <h2 className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Tags</h2>
              <div className="flex flex-wrap">
                <Link to="/tags/next-js" className="mr-3 text-sm font-medium uppercase text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">next-js</Link>
                <Link to="/tags/tailwind" className="mr-3 text-sm font-medium uppercase text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">tailwind</Link>
              </div>
            </div>
            <div className="pt-4 xl:pt-8">
              <Link to="/blog" className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">
                &larr; Back to the blog
              </Link>
            </div>
          </footer>
        </div>
      </div>
    </article>
  );
};

export default BlogDetail;
