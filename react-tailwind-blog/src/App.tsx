import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import BlogList from './pages/BlogList';
import BlogDetail from './pages/BlogDetail';
import About from './pages/About';
import Tags from './pages/Tags';
import Projects from './pages/Projects';
import ProjectDetail from './pages/ProjectDetail';

function App() {
  return (
    <BrowserRouter basename="/AIStudy">
      <Layout>
        <Routes>
          <Route path="/" element={<Projects />} />
          <Route path="/home" element={<Home />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
          <Route path="/blog" element={<BlogList />} />
          <Route path="/blog/:id" element={<BlogDetail />} />
          <Route path="/about" element={<About />} />
          <Route path="/tags" element={<Tags />} />
          <Route path="/tags/:tag" element={<BlogList />} />
          <Route path="/projects" element={<Projects />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
