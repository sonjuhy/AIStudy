import { Link } from 'react-router-dom';
import { useState } from 'react';

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

const BlogList = () => {
  const [searchValue, setSearchValue] = useState('');
  const filteredPosts = MOCK_POSTS.filter((post) => {
    const searchContent = post.title + post.summary + post.tags.join(' ');
    return searchContent.toLowerCase().includes(searchValue.toLowerCase());
  });

  return (
    <div className="divide-y divide-gray-200 dark:divide-gray-700">
      <div className="space-y-2 pb-8 pt-6 md:space-y-5">
        <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:text-6xl md:leading-14">
          All Posts
        </h1>
        <div className="relative max-w-lg">
          <input
            aria-label="Search articles"
            type="text"
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder="Search articles"
            className="block w-full rounded-md border border-gray-300 bg-white px-4 py-2 text-gray-900 focus:border-primary-500 focus:ring-primary-500 dark:border-gray-900 dark:bg-gray-800 dark:text-gray-100"
          />
          <svg
            className="absolute right-3 top-3 h-5 w-5 text-gray-400 dark:text-gray-300"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>
      <ul>
        {!filteredPosts.length && 'No posts found.'}
        {filteredPosts.map((post) => {
          const { slug, date, title, summary, tags } = post;
          return (
            <li key={slug} className="py-4">
              <article className="space-y-2 xl:grid xl:grid-cols-4 xl:items-baseline xl:space-y-0">
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
                <div className="space-y-3 xl:col-span-3">
                  <div>
                    <h3 className="text-2xl font-bold leading-8 tracking-tight">
                      <Link to={`/blog/${slug}`} className="text-gray-900 dark:text-gray-100">
                        {title}
                      </Link>
                    </h3>
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
              </article>
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default BlogList;
