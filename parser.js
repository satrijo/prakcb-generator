import fs from 'fs';
import path from 'path';

const data = fs.readFileSync(path.join('CB','20102025','prakiraancb_20251020182609.json'), 'utf8');
const json = JSON.parse(data);

console.log(json);

const content = json.content;
const contentJson = JSON.parse(content);

console.log(contentJson);

const title = contentJson.cover.title;
const date = contentJson.cover.date;
const image = contentJson.cover.image;
const ocnl = contentJson.cover.ocnl;
const frq = contentJson.cover.frq;

console.log(title, date, image, ocnl, frq);