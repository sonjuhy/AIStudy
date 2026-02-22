const Tags = () => {
  const tagsData = {
    'next-js': 5,
    'tailwind': 4,
    'guide': 3,
    'feature': 2,
    'multi-author': 1,
  };

  return (
    <>
      <div className="flex flex-col items-start justify-start divide-y divide-gray-200 dark:divide-gray-700 md:mt-24 md:flex-row md:items-center md:justify-center md:space-x-6 md:divide-y-0">
        <div className="space-x-2 pb-8 pt-6 md:space-y-5">
          <h1 className="text-3xl font-extrabold leading-9 tracking-tight text-gray-900 dark:text-gray-100 sm:text-4xl sm:leading-10 md:border-r-2 md:px-6 md:text-6xl md:leading-14">
            Tags
          </h1>
        </div>
        <div className="flex max-w-lg flex-wrap">
          {Object.keys(tagsData).length === 0 && 'No tags found.'}
          {Object.entries(tagsData).map(([tag, count]) => {
            return (
              <div key={tag} className="mb-2 mr-5 mt-2">
                <a href={`/tags/${tag}`} className="mr-3 text-sm font-medium uppercase text-primary-500 hover:text-primary-600 dark:hover:text-primary-400">
                  {tag}
                </a>
                <a href={`/tags/${tag}`} className="-ml-2 text-sm font-semibold uppercase text-gray-600 dark:text-gray-300" aria-label={`View posts tagged ${tag}`}>
                  {` (${count})`}
                </a>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
};

export default Tags;
