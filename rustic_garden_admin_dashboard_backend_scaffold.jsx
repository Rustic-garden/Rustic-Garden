# Rustic Garden — Admin Dashboard & Backend Scaffold

This single-file package contains: a brief README, a Node/Express backend scaffold (server.js), a sample SQLite DB migration, and a single-file React admin dashboard (AdminApp.jsx) that you can preview locally. The admin UI uses Tailwind utility classes (no CDN import required in the scaffold instructions) and fetches data from the Express API. It also contains sample integration points for Twilio (SMS), Cloudinary (image uploads), and GitHub content commits.

---

## README (quickstart)

1. Prerequisites: Node 18+, npm, Git.
2. Copy this scaffold into a new folder (e.g., `rustic-backend`).
3. `cd rustic-backend` then `npm install`.
4. Create a `.env` file with required secrets (example below).
5. Run `npm run dev` to start the backend and `npm run admin` to start the React admin locally.
6. Set up hosting: use Render/Heroku/ Railway for backend; host admin on Vercel/Netlify or serve as a GitHub Pages static build.

Example `.env`:

```
PORT=4000
DATABASE_URL=sqlite://./data/db.sqlite
CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=yyyyyyyyyyyyyyyyyyyy
TWILIO_FROM=+1234567890
GITHUB_TOKEN=ghp_xxx...   # optional: use to commit content back to your repo
FRONTEND_URL=http://localhost:5173
```

---

## File: package.json

```json
{
  "name": "rustic-backend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js --watch ./",
    "admin": "vite --config admin.vite.config.js",
    "build-admin": "vite build --config admin.vite.config.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "body-parser": "^1.20.2",
    "cors": "^2.8.5",
    "sqlite3": "^5.1.6",
    "knex": "^2.5.1",
    "multer": "^1.4.5-lts.1",
    "cloudinary": "^1.32.0",
    "twilio": "^4.12.0",
    "node-fetch": "^3.3.2",
    "dotenv": "^16.0.3",
    "passport": "^0.6.0",
    "passport-local": "^1.0.0",
    "bcrypt": "^5.1.0"
  },
  "devDependencies": {
    "nodemon": "^2.0.22",
    "vite": "^5.0.0"
  }
}
```

---

## File: server.js (Express backend)

```js
import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { initDb, db } from './db.js';
import { v2 as cloudinary } from 'cloudinary';
import Twilio from 'twilio';

dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors({ origin: process.env.FRONTEND_URL || '*' }));
app.use(bodyParser.json());

// Init DB
await initDb();

// Configure Cloudinary
if (process.env.CLOUDINARY_URL) cloudinary.config({ secure: true });

// Twilio client
const twClient = process.env.TWILIO_ACCOUNT_SID ? Twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN) : null;

// Multer for local file handling (optional)
const storage = multer.memoryStorage();
const upload = multer({ storage });

// --- API Endpoints ---

// Health
app.get('/api/health', (req, res) => res.json({ ok: true, time: Date.now() }));

// Products CRUD
app.get('/api/products', async (req, res) => {
  const products = await db('products').select();
  res.json(products);
});

app.post('/api/products', async (req, res) => {
  const payload = req.body;
  const [id] = await db('products').insert(payload);
  const product = await db('products').where({ id }).first();
  res.json(product);
});

app.put('/api/products/:id', async (req, res) => {
  const id = req.params.id;
  await db('products').where({ id }).update(req.body);
  const product = await db('products').where({ id }).first();
  res.json(product);
});

app.delete('/api/products/:id', async (req, res) => {
  await db('products').where({ id: req.params.id }).del();
  res.json({ deleted: true });
});

// Blog posts
app.get('/api/posts', async (req, res) => {
  const posts = await db('posts').select().orderBy('created_at', 'desc');
  res.json(posts);
});

app.post('/api/posts', async (req, res) => {
  const [id] = await db('posts').insert({ ...req.body, created_at: new Date().toISOString() });
  const post = await db('posts').where({ id }).first();
  res.json(post);
});

// Image upload: upload to Cloudinary
app.post('/api/upload', upload.single('image'), async (req, res) => {
  try {
    if (!process.env.CLOUDINARY_URL) return res.status(500).json({ error: 'Cloudinary not configured' });
    const result = await cloudinary.uploader.upload_stream({ folder: 'rustic-garden' }, (err, data) => {
      // handled below via stream
    });
    // Use a Promise wrapper
    const streamUpload = (fileBuffer) => new Promise((resolve, reject) => {
      const stream = cloudinary.uploader.upload_stream({ folder: 'rustic-garden' }, (error, result) => {
        if (error) return reject(error);
        resolve(result);
      });
      stream.end(fileBuffer);
    });

    const data = await streamUpload(req.file.buffer);
    res.json({ url: data.secure_url });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Place order endpoint (safe server-side processing)
app.post('/api/order', async (req, res) => {
  const order = {
    product: req.body.product,
    quantity: req.body.quantity,
    name: req.body.customerName,
    phone: req.body.customerPhone,
    email: req.body.customerEmail || null,
    address: req.body.deliveryAddress,
    created_at: new Date().toISOString()
  };

  const [id] = await db('orders').insert(order);
  const saved = await db('orders').where({ id }).first();

  // send SMS via Twilio (if configured)
  if (twClient && process.env.TWILIO_FROM) {
    try {
      await twClient.messages.create({
        from: process.env.TWILIO_FROM,
        to: order.phone,
        body: `Thanks ${order.name}! We received your order #RG${id}. We'll call to confirm.`
      });
    } catch (err) {
      console.warn('Twilio send failed', err.message);
    }
  }

  res.json({ ok: true, order: saved });
});

// Start server
const port = process.env.PORT || 4000;
app.listen(port, () => console.log(`Server running on ${port}`));
```

---

## File: db.js (Knex + SQLite initializer)

```js
import knex from 'knex';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

dotenv.config();
const dbFile = process.env.DATABASE_URL ? process.env.DATABASE_URL.replace('sqlite://', '') : './data/db.sqlite';
const dir = path.dirname(dbFile);
if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

export const db = knex({
  client: 'sqlite3',
  connection: { filename: dbFile },
  useNullAsDefault: true
});

export async function initDb() {
  // Create tables if not exist
  if (!(await db.schema.hasTable('products'))) {
    await db.schema.createTable('products', (t) => {
      t.increments('id').primary();
      t.string('title');
      t.text('description');
      t.integer('price');
      t.string('image_url');
      t.string('slug');
      t.timestamps(true, true);
    });
  }

  if (!(await db.schema.hasTable('posts'))) {
    await db.schema.createTable('posts', (t) => {
      t.increments('id').primary();
      t.string('title');
      t.text('body');
      t.string('cover_url');
      t.string('slug');
      t.timestamp('created_at');
    });
  }

  if (!(await db.schema.hasTable('orders'))) {
    await db.schema.createTable('orders', (t) => {
      t.increments('id').primary();
      t.string('product');
      t.integer('quantity');
      t.string('name');
      t.string('phone');
      t.string('email');
      t.text('address');
      t.timestamp('created_at');
    });
  }
}
```

---

## File: admin.vite.config.js (for running the React admin quickly)

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 }
});
```

---

## File: AdminApp.jsx (single-file React admin)

> Note: This file assumes a Vite + React setup. It uses fetch to talk to `/api` endpoints. It includes CRUD for products, posts, an image upload widget, site theme controls (colors), and a simple page editor for adding/removing navigation tabs. Auth is intentionally minimal — for production add proper sessions / OAuth.

```jsx
import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';

function Admin() {
  const [products, setProducts] = useState([]);
  const [posts, setPosts] = useState([]);
  const [activeTab, setActiveTab] = useState('products');
  const [form, setForm] = useState({ title: '', description: '', price: 0 });
  const [siteConfig, setSiteConfig] = useState({ primaryColor: '#8b5a3c', accentColor: '#a67c5a', nav: ['home','coffee','story','products','order'] });

  useEffect(() => { fetchAll(); }, []);

  async function fetchAll(){
    const p = await fetch('/api/products').then(r=>r.json());
    const po = await fetch('/api/posts').then(r=>r.json());
    setProducts(p);
    setPosts(po);
  }

  async function addProduct(){
    const res = await fetch('/api/products', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(form) });
    const obj = await res.json();
    setProducts(prev => [obj, ...prev]);
    setForm({ title: '', description: '', price: 0 });
  }

  async function uploadImage(file){
    const fd = new FormData();
    fd.append('image', file);
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    return data.url;
  }

  async function handleFileChange(e){
    const url = await uploadImage(e.target.files[0]);
    alert('Uploaded: ' + url);
  }

  // Simple page/tabs editor
  function addNavItem(){
    const label = prompt('New tab slug (e.g. contact)');
    if(!label) return;
    setSiteConfig(prev => ({ ...prev, nav: [...prev.nav, label] }));
    // Optionally persist to repo via GitHub API or server endpoint
  }

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 20 }}>
      <h1>Rustic Garden — Admin</h1>
      <div style={{ display: 'flex', gap: 20 }}>
        <nav style={{ width: 220 }}>
          <button onClick={()=>setActiveTab('products')}>Products</button>
          <button onClick={()=>setActiveTab('posts')}>Blog Posts</button>
          <button onClick={()=>setActiveTab('site')}>Site</button>
          <button onClick={()=>setActiveTab('orders')}>Orders</button>
        </nav>

        <main style={{ flex: 1 }}>
          {activeTab === 'products' && (
            <section>
              <h2>Products</h2>
              <div style={{ marginBottom: 12 }}>
                <input placeholder="Title" value={form.title} onChange={e=>setForm({...form, title:e.target.value})} />
                <input placeholder="Price" value={form.price} onChange={e=>setForm({...form, price: parseInt(e.target.value||0)})} />
                <input placeholder="Short description" value={form.description} onChange={e=>setForm({...form, description:e.target.value})} />
                <input type="file" onChange={handleFileChange} />
                <button onClick={addProduct}>Add Product</button>
              </div>

              <ul>
                {products.map(p=> (
                  <li key={p.id} style={{ padding: 8, border: '1px solid #eee', marginBottom: 6 }}>
                    <strong>{p.title}</strong> — KSh {p.price}
                    <div>{p.description}</div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {activeTab === 'posts' && (
            <section>
              <h2>Blog Posts</h2>
              <button onClick={async ()=>{
                const title = prompt('Title');
                const body = prompt('Body');
                if(!title||!body) return;
                await fetch('/api/posts', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ title, body }) });
                fetchAll();
              }}>Create new post</button>

              <ul>
                {posts.map(p=> (
                  <li key={p.id}><strong>{p.title}</strong> — {new Date(p.created_at).toLocaleString()}</li>
                ))}
              </ul>
            </section>
          )}

          {activeTab === 'site' && (
            <section>
              <h2>Site Settings</h2>
              <div>
                <label>Primary color</label>
                <input value={siteConfig.primaryColor} onChange={e=>setSiteConfig({...siteConfig, primaryColor:e.target.value})} />
              </div>
              <div>
                <h3>Navigation tabs</h3>
                <ul>
                  {siteConfig.nav.map(s=> <li key={s}>{s}</li>)}
                </ul>
                <button onClick={addNavItem}>Add Tab</button>
              </div>
            </section>
          )}

          {activeTab === 'orders' && (
            <section>
              <h2>Orders</h2>
              <p>Orders are stored in the server DB. You can add a list view endpoint or export CSV.</p>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<Admin />);
```

---

## Deployment & GitHub (notes)

- If your site is currently a static site on GitHub Pages, you can keep the frontend (site) deployed there and host this backend on Render/Heroku/Vercel (server). The admin app can be deployed as a separate static app (Vercel/Netlify) and use the server's `/api` endpoints.

- Option: Use GitHub API to commit content back into the repo (e.g., posts as markdown under `_posts/`), so your website (Jekyll/Hugo) rebuilds on push. This requires creating server endpoints that call the GitHub REST API with a `GITHUB_TOKEN` stored as a secret.

- Image storage: Cloudinary is the fastest route for uploads; you can also store images in an S3 bucket and return URLs.

- Twilio: Use the provided `/api/order` endpoint to send SMS confirmations without exposing credentials client-side.

---

## Security notes

- Never store secrets in client-side code. Keep Twilio, Cloudinary, GitHub tokens on server environment variables.
- Add authentication for the admin (JWT, sessions, OAuth). I included `passport` and `bcrypt` in package.json for an easy username/password flow if you want.

---

## Next steps I can do for you (pick any):
1. Wire up admin authentication (email/password or GitHub OAuth).
2. Add CSV export for orders and an orders list UI.
3. Add GitHub commit flow so blog posts are added as markdown files to your repo and automatically published.
4. Create a deployable ready-to-run repo with built-in Dockerfile and Render/Vercel deploy configs.


---

*End of scaffold.*
