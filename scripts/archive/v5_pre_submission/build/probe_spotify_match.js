#!/usr/bin/env node
// Probe Spotify artist-name match rate for a sample of our catalog artists.

const https = require('https');
const fs = require('fs');
const CLIENT_ID = '33f5a2096dfc46d6a420901dd09e6f01';
const CLIENT_SECRET = 'c63c41a2313943c693004a0ac412bed7';

function req(opts, body=null) {
  return new Promise((resolve, reject) => {
    const r = https.request(opts, res => {
      let d=''; res.on('data', c=>d+=c);
      res.on('end', ()=>{ try{ resolve({status:res.statusCode, body:JSON.parse(d)});}catch(e){reject(new Error('parse '+d.slice(0,200)));}});
    });
    r.on('error', reject); if(body) r.write(body); r.end();
  });
}

let TOKEN=null;
async function token(){
  if(TOKEN) return TOKEN;
  const creds = Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString('base64');
  const body='grant_type=client_credentials';
  const r=await req({hostname:'accounts.spotify.com',path:'/api/token',method:'POST',
    headers:{'Authorization':`Basic ${creds}`,'Content-Type':'application/x-www-form-urlencoded','Content-Length':body.length}},body);
  TOKEN=r.body.access_token; return TOKEN;
}

async function searchArtist(name){
  const t=await token();
  const r=await req({hostname:'api.spotify.com',path:`/v1/search?q=${encodeURIComponent(name)}&type=artist&limit=1`,method:'GET',
    headers:{'Authorization':`Bearer ${t}`}});
  return r.body?.artists?.items?.[0] || null;
}

(async()=>{
  const artists = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  let matched=0, sample=[];
  for (const name of artists) {
    try {
      const a = await searchArtist(name);
      if (a && a.name.toLowerCase() === name.toLowerCase()) {
        matched++;
        if (sample.length<5) sample.push({q:name, hit:a.name, id:a.id, followers:a.followers?.total});
      } else if (a) {
        // fuzzy hit
        if (sample.length<5) sample.push({q:name, hit:a.name+'(fuzzy)', id:a.id});
      }
    } catch(e) {
      console.error('err', name, e.message);
    }
    await new Promise(r=>setTimeout(r, 100));
  }
  console.log(`exact-match: ${matched}/${artists.length} (${(100*matched/artists.length).toFixed(1)}%)`);
  console.log('samples:', JSON.stringify(sample, null, 2));
})();
