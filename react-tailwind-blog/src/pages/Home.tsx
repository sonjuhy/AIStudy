import { Link } from 'react-router-dom';

const MOCK_POSTS = [
  {
    slug: 'release-of-tailwind-nextjs-starter-blog-v2.0',
    date: '2023-08-05',
    title: 'Release of Tailwind Nextjs Starter Blog v2.0',
    summary: 'Release of Tailwind Nextjs Starter Blog v2.0. Refactored with App Router and Contentlayer',
    tags: ['next-js', 'tailwind', 'guide'],
  },
  {
    slug: 'new-features-in-v1',
    date: '2021-08-07',
    title: 'New features in v1',
    summary: 'An overview of the new features released in v1 - code block copy, multiple authors, frontmatter layout and more',
    tags: ['next-js', 'tailwind', 'guide'],
  },
  {
    slug: 'introducing-multi-part-posts-with-nested-routing',
    date: '2021-05-02',
    title: 'Introducing Multi-part Posts with Nested Routing',
    summary: 'The blog template supports posts in nested sub-folders. This can be used to group posts of similar content e.g. a multi-part course.',
    tags: ['multi-author', 'next-js', 'feature'],
  },
  {
    slug: 'introducing-tailwind-nextjs-starter-blog',
    date: '2021-01-12',
    title: 'Introducing Tailwind Nextjs Starter Blog',
    summary: 'Looking for a performant, out of the box template, with all the best in web technology to support your blogging needs? Checkout the Tailwind Nextjs Starter Blog template.',
    tags: ['next-js', 'tailwind', 'guide'],
  },
];

const Home = () => {
  return (
    <>
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        <div className="space-y-2 pb-8 pt-6 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
            Latest
          </h1>
          <p className="text-lg leading-7 text-gray-500 dark:text-gray-400">
            A blog created with React and Tailwind.css
          </p>
        </div>
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {!MOCK_POSTS.length && 'No posts found.'}
          {MOCK_POSTS.map((post) => {
            const { slug, date, title, summary, tags } = post;
            return (
              <li key={slug} className="py-12">
                <article>
                  <div className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
                    <dl>
                      <dt className="sr-only">Published on</dt>
                      <dd className="text-base font-medium leading-6 text-gray-500 dark:text-gray-400">
                        <time dateTime={date}>{new Date(date).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric'
                        })}</time>
                      </dd>
                    </dl>
                    <div className="space-y-5 xl:col-span-3">
                      <div className="space-y-6">
                        <div>
                          <h2 className="text-2xl font-bold leading-8 tracking-tight">
                            <Link to={`/blog/${slug}`} className="text-gray-900 dark:text-gray-100">
                              {title}
                            </Link>
                          </h2>
                          <div className="flex flex-wrap">
                            {tags.map((tag) => (
                              <Link key={tag} to={`/tags/${tag}`} className="mr-3 text-sm font-medium uppercase text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">
                                {tag}
                              </Link>
                            ))}
                          </div>
                        </div>
                        <div className="prose max-w-none text-gray-500 dark:text-gray-400">
                          {summary}
                        </div>
                      </div>
                      <div className="text-base font-medium leading-6">
                        <Link to={`/blog/${slug}`} className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400" aria-label={`Read "${title}"`}>
                          Read more &rarr;
                        </Link>
                      </div>
                    </div>
                  </div>
                </article>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="flex justify-end text-base font-medium leading-6">
        <Link to="/blog" className="text-primary-500 hover:text-primary-600 dark:hover:text-primary-400" aria-label="all posts">
          All Posts &rarr;
        </Link>
      </div>
    </>
  );
};

export default Home;
