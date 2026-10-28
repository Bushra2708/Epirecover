import React, { useState, useEffect } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [script, setScript] = useState('tamil');
  const [inscriptionId, setInscriptionId] = useState('INS-TAMIL-001');
  const [period, setPeriod] = useState('10th century CE (Chola)');
  const [region, setRegion] = useState('Thanjavur, Tamil Nadu');
  const [monument, setMonument] = useState('Brihadisvara Temple');
  const [verifiedTranscription, setVerifiedTranscription] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  
  const [corpusQuery, setCorpusQuery] = useState('கோப்பரகேசரி');
  const [corpusResults, setCorpusResults] = useState([]);
  const [reviewDecisions, setReviewDecisions] = useState([]);
  const [reviewNote, setReviewNote] = useState('');

  // Handle file select
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    }
  };

  // Run pipeline
  const handleProcess = async () => {
    if (!file) {
      alert('Please select an inscription image first.');
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('inscription_id', inscriptionId);
    formData.append('script', script);
    formData.append('historical_period', period);
    formData.append('region', region);
    formData.append('monument', monument);
    if (verifiedTranscription) {
      formData.append('verified_transcription', verifiedTranscription);
    }

    try {
      const res = await fetch('/api/pipeline/process', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`Pipeline error: ${res.statusText}`);
      }
      const data = await res.json();
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle review decision
  const handleReviewDecision = async (candidate, decision) => {
    try {
      const res = await fetch('/api/review/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inscription_id: report?.inscription_id || inscriptionId,
          candidate_id: candidate.candidate_id,
          candidate_text: candidate.candidate_text,
          decision: decision,
          notes: reviewNote,
        }),
      });
      const data = await res.json();
      alert(`Decision recorded: ${decision}`);
      fetchDecisions();
    } catch (err) {
      alert(`Failed to save review: ${err.message}`);
    }
  };

  const fetchDecisions = async () => {
    try {
      const res = await fetch('/api/review/decisions');
      const data = await res.json();
      setReviewDecisions(data.decisions || []);
    } catch (e) {}
  };

  const handleCorpusSearch = async () => {
    try {
      const res = await fetch(`/api/corpus/search?query=${encodeURIComponent(corpusQuery)}`);
      const data = await res.json();
      setCorpusResults(data.records || []);
    } catch (err) {
      alert(err.message);
    }
  };

  useEffect(() => {
    fetchDecisions();
  }, []);

  return (
    <div>
      {/* App Header */}
      <header className="app-header">
        <div>
          <div className="brand-title">
            <span>EpiRecover</span>
          </div>
          <div className="brand-subtitle">
            Evidence-Grounded Multimodal Reconstruction of Damaged Ancient Indian Inscriptions
          </div>
        </div>
        <div className="scope-badge">
          Target Language: Tamil Only (v0.1)
        </div>
      </header>

      {/* Main Content */}
      <main className="main-layout">
        {/* Navigation Tabs */}
        <div className="nav-tabs">
          <button 
            className={`nav-tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            Reconstruction Pipeline
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'review' ? 'active' : ''}`}
            onClick={() => setActiveTab('review')}
          >
            Human Verification & Provenance
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
            onClick={() => setActiveTab('evidence')}
          >
            Corpus Evidence Search
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            D0-D6 Synthetic Benchmark
          </button>
        </div>

        {/* Tab 1: Pipeline */}
        {activeTab === 'pipeline' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {/* Stage A: Input & Metadata */}
            <div className="glass-card">
              <div className="card-title">Stage A: Monument Inscription Image & Epigraphic Metadata</div>
              <div className="grid-2">
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                    Upload Monument Photograph (JPG / PNG)
                  </label>
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange}
                    style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', border: '1px solid var(--border-color)', borderRadius: '6px', width: '100%', color: '#fff' }}
                  />
                  {previewUrl && (
                    <div style={{ marginTop: '1rem', border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', maxHeight: '240px' }}>
                      <img src={previewUrl} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                  <div className="grid-2">
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Target Language</label>
                      <input type="text" value="Tamil (Fixed in v0.1)" disabled style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.4)', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--accent-gold)' }} />
                    </div>
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Script Selection</label>
                      <select 
                        value={script} 
                        onChange={(e) => setScript(e.target.value)}
                        style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }}
                      >
                        <option value="tamil">Modern Tamil Script</option>
                        <option value="tamil-brahmi">Historical Tamil-Brahmi Script</option>
                      </select>
                    </div>
                  </div>

                  {script === 'tamil-brahmi' && (
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.6rem 0.8rem', borderRadius: '6px', fontSize: '0.8rem', color: '#fca5a5' }}>
                      ⚠️ <strong>Historical Script Notice:</strong> Standard Tesseract/EasyOCR models do not recognize Tamil-Brahmi. Ancient glyphs will not be misclassified as modern Tamil; readings will be retrieved from verified corpus records and marked with explicit gap markers.
                    </div>
                  )}

                  <div className="grid-2">
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Inscription ID</label>
                      <input type="text" value={inscriptionId} onChange={(e) => setInscriptionId(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }} />
                    </div>
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Historical Period</label>
                      <input type="text" value={period} onChange={(e) => setPeriod(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }} />
                    </div>
                  </div>

                  <div className="grid-2">
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Region</label>
                      <input type="text" value={region} onChange={(e) => setRegion(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }} />
                    </div>
                    <div>
                      <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Monument / Site</label>
                      <input type="text" value={monument} onChange={(e) => setMonument(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }} />
                    </div>
                  </div>

                  <div>
                    <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Verified Ground Truth Reference (Optional, for CER/WER)</label>
                    <input type="text" value={verifiedTranscription} onChange={(e) => setVerifiedTranscription(e.target.value)} placeholder="e.g. ஸ்வஸ்தி ஸ்ரீ கோப்பரகேசரிவன்மற்கு யாண்டு" style={{ width: '100%', padding: '0.5rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '4px', color: '#fff' }} />
                  </div>

                  <button className="btn btn-primary" onClick={handleProcess} disabled={loading} style={{ alignSelf: 'flex-start', marginTop: '0.5rem' }}>
                    {loading ? 'Processing Stages A to G...' : 'Execute Multimodal Reconstruction'}
                  </button>
                </div>
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', padding: '1rem', borderRadius: '8px', color: '#fca5a5' }}>
                {error}
              </div>
            )}

            {/* Report Output */}
            {report && (
              <>
                {/* Stage C & D: Read the Stone Output */}
                <div className="grid-2">
                  <div className="glass-card">
                    <div className="card-title">Stage C: Damage & Spalling Analysis</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                      Method: <span style={{ color: 'var(--accent-amber)' }}>{report.damage_label || report.damage_method}</span>
                    </div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                      Detected Weathered / Fractured Regions: <strong>{report.damage_regions?.length || 0}</strong>
                    </div>
                  </div>

                  <div className="glass-card">
                    <div className="card-title">Stage D: Line OCR & Explicit Gap Markers</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                      Engine: <strong>{report.ocr_engine}</strong> | Mean Confidence: <strong>{report.ocr_mean_confidence}%</strong>
                    </div>
                    <div className="tamil-text-display">
                      {report.ocr_raw_text ? report.ocr_raw_text : <span className="gap-marker">[UNKNOWN GAPS DETECTED]</span>}
                    </div>
                  </div>
                </div>

                {/* Stage E & G: Candidates & Fusion */}
                <div className="glass-card">
                  <div className="card-title">
                    Stage E & G: Top-K Evidence-Grounded Reconstruction Candidates
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                    Scored via Multimodal Fusion (V + O + L + H + R). Uncalibrated ranking is transparently displayed.
                  </div>

                  {report.should_abstain && (
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.8rem', borderRadius: '6px', marginBottom: '1rem', color: '#fca5a5', fontSize: '0.85rem' }}>
                      🛑 <strong>Abstention Triggered:</strong> {report.abstention_reason} System refuses to hallucinate text.
                    </div>
                  )}

                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Candidate ID</th>
                        <th>Proposed Reading</th>
                        <th>Provenance</th>
                        <th>Score Breakdown [V, O, L, H, R]</th>
                        <th>Rank Score</th>
                        <th>Expert Review</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.candidates?.map((cand) => (
                        <tr key={cand.candidate_id}>
                          <td><code>{cand.candidate_id}</code></td>
                          <td style={{ fontSize: '1rem', color: '#fce09b', fontFamily: 'JetBrains Mono, monospace' }}>
                            {cand.candidate_text}
                          </td>
                          <td>
                            <span className={`prov-badge prov-${cand.provenance}`}>
                              {cand.provenance}
                            </span>
                          </td>
                          <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                            V:{cand.rank_details?.V || 0} | O:{cand.rank_details?.O || 0} | L:{cand.rank_details?.L || 0} | H:{cand.rank_details?.H || 0} | R:{cand.rank_details?.R || 0}
                          </td>
                          <td>
                            <strong style={{ color: 'var(--accent-gold)' }}>{cand.rank_score ?? cand.confidence}</strong>
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: '0.4rem' }}>
                              <button className="btn btn-accept" style={{ padding: '3px 8px', fontSize: '0.75rem' }} onClick={() => handleReviewDecision(cand, 'ACCEPTED')}>Accept</button>
                              <button className="btn btn-reject" style={{ padding: '3px 8px', fontSize: '0.75rem' }} onClick={() => handleReviewDecision(cand, 'REJECTED')}>Reject</button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Evaluation Metrics (if verified transcription was supplied) */}
                {report.evaluation && (
                  <div className="glass-card">
                    <div className="card-title">Quantitative Evaluation (Verified Ground Truth Reference)</div>
                    <div className="grid-3">
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Character Error Rate (CER)</div>
                        <div style={{ fontSize: '1.4rem', color: 'var(--accent-gold)', fontWeight: 700 }}>
                          {report.evaluation.cer !== null ? (report.evaluation.cer * 100).toFixed(2) + '%' : 'Unavailable'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Word Error Rate (WER)</div>
                        <div style={{ fontSize: '1.4rem', color: 'var(--accent-gold)', fontWeight: 700 }}>
                          {report.evaluation.wer !== null ? (report.evaluation.wer * 100).toFixed(2) + '%' : 'Unavailable'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Unicode Normalization</div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-main)', marginTop: '4px' }}>
                          NFC (Canonical Composition)
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Tab 2: Human Review */}
        {activeTab === 'review' && (
          <div className="glass-card">
            <div className="card-title">Human Epigraphist Review Ledger</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Records immutable decisions made by researchers. Machine predictions are never conflated with human-verified transcriptions.
            </div>

            <table className="data-table">
              <thead>
                <tr>
                  <th>Decision ID</th>
                  <th>Inscription ID</th>
                  <th>Candidate ID</th>
                  <th>Reviewed Reading</th>
                  <th>Status</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {reviewDecisions.length === 0 ? (
                  <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dim)' }}>No human reviews recorded yet.</td></tr>
                ) : (
                  reviewDecisions.map((dec, i) => (
                    <tr key={i}>
                      <td><code>{dec.decision_id}</code></td>
                      <td>{dec.inscription_id}</td>
                      <td><code>{dec.candidate_id}</code></td>
                      <td style={{ color: '#fce09b', fontFamily: 'monospace' }}>{dec.candidate_text}</td>
                      <td>
                        <span className={`prov-badge ${dec.decision.toLowerCase() === 'accepted' ? 'prov-observed' : 'prov-unknown'}`}>
                          {dec.decision}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{dec.timestamp}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Evidence Search */}
        {activeTab === 'evidence' && (
          <div className="glass-card">
            <div className="card-title">Historical Tamil Inscription Evidence Corpus</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Queries verified lithic inscriptions from BharatSHRI (ASI) and SIDDHAM repositories.
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
              <input 
                type="text" 
                value={corpusQuery} 
                onChange={(e) => setCorpusQuery(e.target.value)} 
                placeholder="Search Tamil phrase or ruler name..."
                style={{ flex: 1, padding: '0.6rem 1rem', background: '#12161f', border: '1px solid var(--border-color)', borderRadius: '6px', color: '#fff' }}
              />
              <button className="btn btn-primary" onClick={handleCorpusSearch}>Search Corpus</button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {corpusResults.map((rec, i) => (
                <div key={i} style={{ background: 'rgba(10,12,16,0.6)', border: '1px solid var(--border-color)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <strong>{rec.id}</strong>
                    <span style={{ fontSize: '0.78rem', color: 'var(--accent-gold)' }}>Score: {rec.score}</span>
                  </div>
                  <div style={{ fontSize: '1.05rem', color: '#fce09b', fontFamily: 'monospace', marginBottom: '0.4rem' }}>
                    {rec.transcription}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {rec.monument} • {rec.region} • {rec.historical_period}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                    Citation: {rec.source_citation}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 4: Benchmark D0-D6 */}
        {activeTab === 'benchmark' && (
          <div className="glass-card">
            <div className="card-title">Synthetic Damage Benchmark (D0 to D6)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Reproducible mathematical damage operations at 10% to 60% degradation (PPT Slides 10 & 14).
            </div>
            <div className="grid-3">
              {[
                { lvl: 'D0', desc: 'Clean unperturbed baseline (0% damage)' },
                { lvl: 'D1', desc: 'Light superficial stone erosion & stroke thinning (~10%)' },
                { lvl: 'D2', desc: 'Structural fractures, fissures, and crack lines (~20%)' },
                { lvl: 'D3', desc: 'Surface spalling & flaking masks (~30%)' },
                { lvl: 'D4', desc: 'Severe missing stone chunks & spalling (~40-50%)' },
                { lvl: 'D5', desc: 'Oblique lighting, harsh shadow & blur (~50%)' },
                { lvl: 'D6', desc: 'Composite extreme degradation (~60%)' },
              ].map((item) => (
                <div key={item.lvl} style={{ background: 'rgba(18,22,31,0.7)', border: '1px solid var(--border-color)', padding: '1rem', borderRadius: '8px' }}>
                  <div style={{ color: 'var(--accent-gold)', fontWeight: 700, fontSize: '1.1rem' }}>{item.lvl}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>{item.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
