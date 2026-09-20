import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { api, ApiError } from './api'
import { searchEntities } from './search'
import type { Bootstrap, Context, Entity, ModelRow, Result, Role, Session, Side, Stat } from './types'

const ROLES: Role[] = ['TOP', 'JUNGLE', 'MID', 'BOT', 'SUPPORT']
const ROLE_LABEL: Record<Role, string> = { TOP: 'Đường trên', JUNGLE: 'Đi rừng', MID: 'Đường giữa', BOT: 'Xạ thủ', SUPPORT: 'Hỗ trợ' }
const MODEL_NAMES: Record<string, string> = { baseline: 'Baseline', logistic_regression: 'Logistic Regression', random_forest: 'Random Forest' }
const emptyContext = (): Context => ({ blue_team_id: '', red_team_id: '', patch: '', context_label: '', user_confirmed: false,
  blue_roster: ROLES.map(role => ({ role, player_id: '' })), red_roster: ROLES.map(role => ({ role, player_id: '' })) })
const pct = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const date = (value: string | null | undefined) => value ? new Date(value).toLocaleDateString('vi-VN') : 'Chưa ghi nhận'
const message = (error: unknown) => error instanceof Error ? error.message : 'Có lỗi xảy ra. Hãy thử lại.'

async function initialize() {
  const data = api<Bootstrap>('/bootstrap')
  // Attach a rejection handler immediately while session restoration is in flight.
  const sessionTask = (async () => {
    const token = sessionStorage.getItem('match-insight-session')
    let session: Session
    if (token) {
      try { session = await api<Session>('/session', 'GET', undefined, token) }
      catch (error) {
        if (!(error instanceof ApiError) || error.code !== 'E_SESSION_EXPIRED') throw error
        session = await api<Session>('/sessions', 'POST')
      }
    } else session = await api<Session>('/sessions', 'POST')
    sessionStorage.setItem('match-insight-session', session.session_id)
    return session
  })()
  const [bootstrap, session] = await Promise.all([data, sessionTask])
  return { bootstrap, session }
}
let initialRequest: ReturnType<typeof initialize> | null = null

function Avatar({ entity, small = false }: { entity?: Entity; small?: boolean }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [entity?.image])
  return <span className={`avatar ${small ? 'small' : ''}`}>
    {entity?.image && !failed ? <img src={entity.image} alt="" loading="lazy" onError={() => setFailed(true)} /> : <span>{entity?.name.slice(0, 2).toUpperCase() ?? '—'}</span>}
  </span>
}

function SearchSelect({ label, options, value, onChange, disabled = false, placeholder = 'Tìm và chọn…', exclude = [] }: {
  label: string; options: Entity[]; value: string; onChange: (id: string) => void; disabled?: boolean; placeholder?: string; exclude?: string[]
}) {
  const uid = useId()
  const selected = options.find(option => option.id === value)
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [index, setIndex] = useState(0)
  const items = useMemo(() => searchEntities(options, query, exclude), [options, exclude, query])
  const choose = (entity: Entity) => { onChange(entity.id); setOpen(false); setQuery('') }
  return <div className="search-select">
    <label htmlFor={uid}>{label}</label>
    <div className="search-input-wrap">
      <input id={uid} role="combobox" aria-expanded={open} aria-controls={`${uid}-list`} aria-autocomplete="list"
        aria-activedescendant={open && items[index] ? `${uid}-${index}` : undefined} autoComplete="off"
        value={open ? query : selected?.name ?? ''} placeholder={placeholder} disabled={disabled}
        onFocus={() => { setQuery(''); setIndex(0); setOpen(true) }}
        onBlur={() => setOpen(false)} onChange={event => { setQuery(event.target.value); setIndex(0); setOpen(true) }}
        onKeyDown={event => {
          if (event.key === 'Escape') setOpen(false)
          if (event.key === 'ArrowDown') { event.preventDefault(); setOpen(true); setIndex(i => Math.min(i + 1, items.length - 1)) }
          if (event.key === 'ArrowUp') { event.preventDefault(); setIndex(i => Math.max(i - 1, 0)) }
          if (event.key === 'Enter' && open) { event.preventDefault(); if (items[index]) choose(items[index]) }
        }} />
      <span aria-hidden="true">⌄</span>
    </div>
    {open && !disabled && <div className="select-options" id={`${uid}-list`} role="listbox" aria-label={label}>
      {items.length ? items.map((entity, i) => <button type="button" role="option" aria-selected={value === entity.id}
        id={`${uid}-${i}`} key={entity.id} className={index === i ? 'focused' : ''}
        onMouseDown={event => event.preventDefault()} onClick={() => choose(entity)}>
        <Avatar entity={entity} small /><span>{entity.name}</span>{entity.history_count != null && <small>{entity.history_count} ván</small>}
      </button>) : <div className="no-options">Không tìm thấy trong phạm vi đang chọn.</div>}
      {items.length === 40 && <small className="search-hint">Nhập thêm tên để thu hẹp kết quả.</small>}
    </div>}
  </div>
}

function ProbabilityBar({ row, phase, side }: { row: ModelRow; phase: 'pre' | 'post'; side: Side }) {
  const value = row[phase]?.[`${side}_win_probability`]
  if (value == null) return <div className="awaiting">{phase === 'post' ? 'Chờ đội hình' : 'Chưa ghi nhận'}</div>
  return <div className="probability"><span>{pct(value)}</span><div className="track" aria-hidden="true"><i className={side} style={{ width: `${value * 100}%` }} /></div></div>
}

function Comparison({ result, selectedFamily }: { result: Result; selectedFamily?: string }) {
  const [side, setSide] = useState<Side>('blue')
  const team = result.teams[side]
  return <section className="panel results-panel" aria-label="Kết quả ba mô hình">
    <div className="panel-heading"><div><span className="eyebrow">{result.phase === 'POST' ? 'PRE → POST' : 'MỐC NỀN · PRE'}</span>
      <h2>Ba mô hình, cùng một ván</h2><p>Xác suất thắng ước lượng của <strong>{team.name}</strong>.</p></div>
      <div className="segmented" aria-label="Đội được so sánh">
        {(['blue', 'red'] as Side[]).map(value => <button key={value} aria-pressed={side === value} onClick={() => setSide(value)}>{result.teams[value].name}</button>)}
      </div>
    </div>
    <div className="table-scroll"><table className="comparison-table"><thead><tr><th>Mô hình</th><th>Trước đội hình <small>PRE</small></th><th>Sau đội hình <small>POST</small></th><th>Thay đổi <small>điểm %</small></th></tr></thead>
      <tbody>{result.models.map(row => <tr key={row.family} className={row.family === selectedFamily ? 'selected-model' : ''}>
        <th><span className="model-name">{row.label}</span><small>{row.family === 'baseline' ? 'Tỷ lệ nền trong tập train' : row.family === selectedFamily ? 'Được chọn trên validation' : 'Mô hình đối chiếu'}</small></th>
        <td data-label="PRE"><ProbabilityBar row={row} phase="pre" side={side} /></td><td data-label="POST"><ProbabilityBar row={row} phase="post" side={side} /></td>
        <td data-label="Δ · điểm %"><span className={`delta ${row.delta ? row.delta[side] >= 0 ? 'positive' : 'negative' : ''}`}>{row.delta ? `${row.delta[side] > 0 ? '+' : ''}${row.delta[side].toFixed(1)}` : '—'}</span></td>
      </tr>)}</tbody></table></div>
    <div className="result-footnote"><span className="info-symbol">i</span><p>Xác suất cao ở một ván không chứng minh mô hình tốt hơn. Chênh lệch PRE–POST thể hiện thay đổi ước lượng khi bổ sung đội hình, không phải tác động nhân quả của draft.</p></div>
  </section>
}

function statText(stat: Stat) {
  if (stat.missing || stat.value === null) return `Chưa ghi nhận · ${stat.sample_count} ván`
  return `${pct(stat.value)} · ${stat.win_count == null ? `${stat.sample_count} ván đối chiếu` : `${stat.win_count}/${stat.sample_count} ván thắng`}`
}

function Evidence({ result }: { result: Result }) {
  const coverage = result.coverage
  return <div className="evidence">
    {coverage && <div className="coverage-strip"><span className="status-dot amber" /><p><strong>{coverage.pre_used.game_count} ván lịch sử được dùng cho PRE</strong><span>Mới nhất: {date(coverage.pre_used.latest_ended_at)} · Mốc cắt: {date(result.history_cutoff_at)}</span></p></div>}
    {result.warnings.length > 0 && <div className="warning-list" role="status">{result.warnings.map((warning, i) => <div key={`${warning.warning_code}-${i}`}>
      <p><span>!</span>{warning.message}</p>
      {warning.warning_code === 'W_PAIR_HISTORY_MISSING' && <ul className="missing-pairs">{result.player_champion?.filter(pair => pair.games_count === 0).map(pair => {
        const side = pair.side.toLowerCase() as Side
        const slot = result.roster?.[side].find(row => row.role === pair.role)
        return <li key={`${pair.side}-${pair.role}`}><strong>{slot?.player.name ?? 'Chưa rõ tuyển thủ'} — {slot?.champion?.name ?? 'Chưa rõ tướng'}</strong><span>{result.teams[side].name} · {ROLE_LABEL[pair.role]} · Chưa ghi nhận ván nào trong phạm vi lịch sử đang dùng.</span></li>
      })}</ul>}
    </div>)}</div>}
    <details className="disclosure"><summary>Dữ liệu lịch sử và phạm vi áp dụng <span>＋</span></summary><div className="disclosure-body">
      <p className="muted">Các tỷ lệ dưới đây là thống kê lịch sử, khác với xác suất dự đoán của mô hình. Dữ liệu không xác nhận phong độ hiện tại.</p>
      {result.statistics && <div className="history-columns">{(['blue', 'red'] as Side[]).map(side => <div key={side}><h3>{result.teams[side].name}</h3>
        {(['recent_form', 'side_win_rate', 'roster_continuity'] as const).map((key, i) => <div className="stat-line" key={key}><span>{['Phong độ trong lịch sử', 'Thắng theo bên thi đấu', 'Liên tục lực lượng'][i]}</span><strong>{statText(result.statistics![side][key])}</strong></div>)}
      </div>)}</div>}
      {result.statistics && <p>Đối đầu của {result.teams.blue.name}: {statText(result.statistics.head_to_head)}</p>}
      {coverage && <><p>Patch {result.patch}: {coverage.patch_observed_in_history ? 'có ghi nhận trong kho lịch sử' : 'chưa ghi nhận trong kho lịch sử'}. Phạm vi hỗ trợ patch chưa được xác minh.</p>
        {coverage.teams.map(team => <p key={team.team_id}>{Object.values(result.teams).find(t => t.id === team.team_id)?.name}: {team.pre_used.game_count} ván thực dùng · mới nhất {date(team.pre_used.latest_ended_at)}.</p>)}</>}
      {!!result.player_champion?.length && <div className="table-scroll"><table><thead><tr><th>Đội / vị trí</th><th>Tuyển thủ · tướng</th><th>Lịch sử chuyên nghiệp</th></tr></thead><tbody>{result.player_champion.map(pair => {
        const side = pair.side.toLowerCase() as Side
        const slot = result.roster?.[side].find(slot => slot.role === pair.role)
        return <tr key={`${pair.side}-${pair.role}`}><td>{result.teams[side].name}<small>{ROLE_LABEL[pair.role]}</small></td><td>{slot?.player.name} · {slot?.champion?.name}</td><td>{pair.win_rate == null ? 'Chưa ghi nhận' : pct(pair.win_rate)}<small>{pair.wins_count}/{pair.games_count} ván thắng</small></td></tr>
      })}</tbody></table></div>}
    </div></details>
    {result.metadata && <details className="disclosure"><summary>Thông tin bản lưu và thời gian xử lý <span>＋</span></summary><div className="disclosure-body technical">
      <p>Mã PRE: {result.pre_evaluation_id} · Mã POST: {result.post_evaluation_id ?? 'Chưa tạo'}</p>
      {result.timing_ms && <p>Phân tích: {result.timing_ms.analysis.toLocaleString('vi-VN')} ms · Chuẩn bị chi tiết: {result.timing_ms.details.toLocaleString('vi-VN')} ms · Lưu: {result.timing_ms.save.toLocaleString('vi-VN')} ms. Số đo tại máy chủ, không gồm truyền mạng và vẽ giao diện.</p>}
      {Object.entries(result.metadata).map(([key, value]) => <p key={key}><strong>{key}</strong><br />{value}</p>)}
    </div></details>}
  </div>
}

function Quality({ data }: { data: Bootstrap }) {
  const metrics = data.metrics
  return <details className="disclosure quality"><summary>Chất lượng ba mô hình trên dữ liệu đánh giá <span>＋</span></summary><div className="disclosure-body">
    <div className="section-intro"><div><h3>Kết quả trên tập validation</h3><p>{metrics.count.toLocaleString('vi-VN')} ván · Cùng phép chia theo thời gian · Không phải kết quả của ván đang chọn.</p></div><span className="tag">Dữ liệu từ mô hình đã lưu</span></div>
    <div className="table-scroll"><table><thead><tr><th>Mô hình</th><th>Mốc</th><th>Brier ↓</th><th>Log Loss ↓</th><th>ROC-AUC ↑</th></tr></thead><tbody>{metrics.scores.flatMap(row => (['pre', 'post'] as const).map(phase => <tr key={`${row.family}-${phase}`}><th>{MODEL_NAMES[row.family]}</th><td>{phase.toUpperCase()}</td><td>{row[phase].brier_score.toFixed(5)}</td><td>{row[phase].log_loss.toFixed(5)}</td><td>{row[phase].roc_auc?.toFixed(5) ?? 'Không áp dụng'}</td></tr>))}</tbody></table></div>
    <p className="muted">Brier và Log Loss càng thấp càng tốt; ROC-AUC càng cao càng tốt. Mô hình được chọn dựa trên Brier trung bình PRE/POST và quy tắc lựa chọn đã cố định. POST không mặc nhiên tốt hơn PRE.</p>
  </div></details>
}

function SavedView({ data }: { data: Bootstrap | null }) {
  const [id, setId] = useState('')
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = async () => { setBusy(true); setError(''); setResult(null); try { setResult(await api<Result>(`/evaluations/${id}`)) } catch (error) { setError(message(error)) } finally { setBusy(false) } }
  return <><div className="page-intro"><span className="eyebrow">LỊCH SỬ PHÂN TÍCH</span><h1>Xem lại kết quả đã lưu</h1><p>Tra cứu bản đánh giá bất biến bằng mã PRE hoặc POST, kể cả sau khi khởi động lại máy chủ.</p></div>
    <form className="panel lookup-form" onSubmit={event => { event.preventDefault(); void load() }}><label htmlFor="evaluation-id">Mã đánh giá<input id="evaluation-id" type="number" min="1" step="1" required value={id} onChange={event => { setId(event.target.value); setResult(null); setError('') }} placeholder="Ví dụ: mã PRE bạn đã lưu" disabled={busy} /></label><button className="primary" disabled={busy || !id}>{busy ? 'Đang đọc…' : 'Tra cứu bản lưu →'}</button></form>
    {error && <div className="error-banner" role="alert">{error}</div>}
    {result && <><div className="saved-heading"><span className="tag">{result.phase} #{result.evaluation_id}</span><span>{result.active ? 'Còn hiệu lực tại thời điểm đọc' : 'Đã mất hiệu lực · giữ để đối chiếu'}</span><span>Patch {result.patch}</span></div><Comparison result={result} selectedFamily={data?.metrics.selected_family} /><Evidence result={result} /><p className="muted">Bản lưu chỉ dùng để xem lại; để phân tích tiếp, tạo PRE trong phiên hiện tại.</p></>}
  </>
}

export default function App() {
  const [page, setPage] = useState<'analysis' | 'saved'>('analysis')
  const [data, setData] = useState<Bootstrap | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [context, setContext] = useState<Context>(emptyContext)
  const [champions, setChampions] = useState<string[]>(Array(10).fill(''))
  const [linkedOnly, setLinkedOnly] = useState(true)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [loadError, setLoadError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const preRequest = useRef<NonNullable<Session['pending_pre']> | null>(null)
  const postRequest = useRef<NonNullable<Session['pending_post']> | null>(null)
  const resultAnchor = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let current = true
    setLoading(true); setLoadError('')
    initialRequest ??= initialize()
    initialRequest.then(({ bootstrap, session: loaded }) => {
      if (!current) return
      setData(bootstrap); setSession(loaded)
      preRequest.current = loaded.pending_pre ?? null
      postRequest.current = loaded.pending_post ?? null
      if (loaded.pre?.roster) setContext({ blue_team_id: loaded.pre.teams.blue.id, red_team_id: loaded.pre.teams.red.id,
        blue_roster: loaded.pre.roster.blue.map(slot => ({ role: slot.role, player_id: slot.player.id })),
        red_roster: loaded.pre.roster.red.map(slot => ({ role: slot.role, player_id: slot.player.id })),
        patch: loaded.pre.patch, context_label: loaded.pre.context_label ?? '', user_confirmed: true })
      else if (loaded.pending_pre) setContext(loaded.pending_pre.context)
      if (loaded.post?.roster) setChampions([...loaded.post.roster.blue, ...loaded.post.roster.red].map(slot => slot.champion?.id ?? ''))
      else if (loaded.pending_post) setChampions(loaded.pending_post.champion_ids)
    }).catch(error => { if (current) setLoadError(message(error)) }).finally(() => { if (current) setLoading(false) })
    return () => { current = false }
  }, [attempt])

  const pre = session?.pre
  const post = session?.post
  const result = post ?? pre
  const contextLocked = !!pre || !!busy || !!preRequest.current
  const lineupLocked = !!post || !!busy || !!postRequest.current
  const teamOptions = useMemo(() => data?.teams.filter(team => !linkedOnly || (team.history_count ?? 0) > 0 || team.id === context.blue_team_id || team.id === context.red_team_id) ?? [], [data, linkedOnly, context.blue_team_id, context.red_team_id])
  const playerOptions = useMemo(() => {
    const selected = new Set([...context.blue_roster, ...context.red_roster].map(slot => slot.player_id))
    return data?.players.filter(player => !linkedOnly || (player.history_count ?? 0) > 0 || selected.has(player.id)) ?? []
  }, [data, linkedOnly, context.blue_roster, context.red_roster])
  const playerIds = [...context.blue_roster, ...context.red_roster].map(slot => slot.player_id)
  const contextComplete = !!context.blue_team_id && !!context.red_team_id && context.blue_team_id !== context.red_team_id && playerIds.every(Boolean) && new Set(playerIds).size === 10 && !!context.patch.trim() && context.user_confirmed
  const lineupComplete = champions.every(Boolean) && new Set(champions).size === 10
  const changeTeam = (side: Side, id: string) => {
    setContext(previous => ({ ...previous, [`${side}_team_id`]: id, [`${side}_roster`]: data?.rosters[id]?.slots ?? ROLES.map(role => ({ role, player_id: '' })), user_confirmed: false }))
    setError('')
  }
  const act = async (label: string, action: () => Promise<void>) => {
    if (busy) return
    setBusy(label); setError('')
    try { await action() } catch (error) { setError(message(error)) } finally { setBusy('') }
  }
  const createPre = () => act('Đang tổng hợp lịch sử, dự đoán và lưu PRE…', async () => {
    if (!session) return
    preRequest.current ??= { operation_id: crypto.randomUUID(), context }
    const value = await api<Result>('/pre', 'POST', preRequest.current, session.session_id)
    preRequest.current = null
    setSession({ ...session, pre: value, post: null }); setChampions(Array(10).fill(''))
    requestAnimationFrame(() => resultAnchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  })
  const createPost = () => act('Đang bổ sung lịch sử tuyển thủ–tướng và lưu POST…', async () => {
    if (!session || !pre?.pre_evaluation_id) return
    postRequest.current ??= { operation_id: crypto.randomUUID(), pre_evaluation_id: pre.pre_evaluation_id, champion_ids: champions }
    const value = await api<Result>('/post', 'POST', postRequest.current, session.session_id)
    postRequest.current = null
    setSession({ ...session, post: value })
    requestAnimationFrame(() => resultAnchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  })
  const edit = (postOnly: boolean) => act('Đang cập nhật trạng thái bản lưu…', async () => {
    if (!session) return
    const next = await api<Session>(postOnly ? '/post' : '/pre', 'DELETE', undefined, session.session_id)
    setSession(next); postRequest.current = null
    if (!postOnly) { preRequest.current = null; setContext(value => ({ ...value, user_confirmed: false })); setChampions(Array(10).fill('')) }
  })

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#" onClick={event => { event.preventDefault(); setPage('analysis') }} aria-label="Match Insight — trang phân tích"><span className="brand-mark"><i /><i /><i /></span><span>MATCH<span className="brand-light">INSIGHT</span></span></a>
      <nav aria-label="Điều hướng"><button className={page === 'analysis' ? 'active' : ''} onClick={() => setPage('analysis')}>Phân tích ván đấu</button><button className={page === 'saved' ? 'active' : ''} onClick={() => setPage('saved')}>Bản đã lưu</button></nav>
      <span className="connection"><span className={`status-dot ${data ? '' : 'amber'}`} />{data ? 'Dữ liệu đã sẵn sàng' : 'Đang kết nối'}</span>
    </header>
    <main>
      {page === 'saved' ? <SavedView data={data} /> : <>
        <div className="page-intro"><span className="eyebrow">LEAGUE OF LEGENDS · PHÂN TÍCH TRƯỚC TRẬN</span><h1>Một ván đấu. Hai mốc đánh giá.</h1><p>So sánh tương quan hai đội trước và sau khi xác nhận đội hình tướng.</p></div>
        <div className="workflow" aria-label="Tiến trình phân tích"><div className={!pre ? 'current' : 'done'}><b>{pre ? '✓' : '01'}</b><span>Xác nhận bối cảnh</span></div><i /><div className={pre && !post ? 'current' : post ? 'done' : ''}><b>{post ? '✓' : '02'}</b><span>Xem PRE · Chọn tướng</span></div><i /><div className={post ? 'current' : ''}><b>03</b><span>So sánh PRE / POST</span></div></div>
        {loading && <div className="panel loading-state" role="status"><span className="spinner" /><h2>Đang chuẩn bị dữ liệu và ba mô hình</h2><p>Lần mở đầu tiên cần đọc kho lịch sử. Các thao tác sau dùng lại tài nguyên đã nạp.</p><div className="skeletons"><i /><i /><i /></div></div>}
        {loadError && <div className="panel loading-state"><h2>Chưa kết nối được dữ liệu</h2><p role="alert">{loadError}</p><button className="primary" onClick={() => { initialRequest = null; setAttempt(value => value + 1) }}>Thử nạp lại</button></div>}
        {!loading && !loadError && data && <>
          <section className="panel context-panel" aria-label="Bối cảnh ván đấu"><div className="panel-heading"><div><span className="eyebrow">01 / BỐI CẢNH</span><h2>Hai đội trên cùng một bản đồ</h2></div>
            {pre || preRequest.current ? <button className="secondary" disabled={!!busy} onClick={() => void edit(false)}>{pre ? 'Chỉnh sửa bối cảnh' : 'Hủy yêu cầu PRE'}</button> : <label className="scope-toggle"><input type="checkbox" checked={linkedOnly} disabled={contextLocked} onChange={event => setLinkedOnly(event.target.checked)} />Chỉ hiện tên có lịch sử</label>}
          </div>
          <div className="context-fields"><label>Patch<input value={context.patch} maxLength={30} disabled={contextLocked} placeholder="Nhập phiên bản thi đấu" onChange={event => setContext(value => ({ ...value, patch: event.target.value, user_confirmed: false }))} /></label><label>Giải / bối cảnh ván <span className="optional">tùy chọn</span><input value={context.context_label} maxLength={160} disabled={contextLocked} placeholder="Tên giải hoặc ghi chú để nhận diện ván" onChange={event => setContext(value => ({ ...value, context_label: event.target.value, user_confirmed: false }))} /></label></div>
          <div className="teams-grid">{(['blue', 'red'] as Side[]).map(side => {
            const team = data.teams.find(row => row.id === context[`${side}_team_id`])
            return <div className={`team-card ${side}`} key={side}><div className="team-title"><Avatar entity={team} /><div><span className="side-label">{side === 'blue' ? 'BÊN XANH' : 'BÊN ĐỎ'}</span><h3>{team?.name ?? 'Chọn đội thi đấu'}</h3></div><span className="side-number">{side === 'blue' ? 'A' : 'B'}</span></div>
              <SearchSelect label={`Đội ${side === 'blue' ? 'bên xanh' : 'bên đỏ'}`} options={teamOptions} value={context[`${side}_team_id`]} onChange={id => changeTeam(side, id)} disabled={contextLocked} exclude={[context[`${side === 'blue' ? 'red' : 'blue'}_team_id`]]} placeholder="Tìm tên đội…" />
              <div className="roster-list">{context[`${side}_roster`].map((slot, index) => <div className="roster-slot" key={slot.role}><span className="role-index">0{index + 1}</span><div className="player-portrait"><Avatar entity={playerOptions.find(player => player.id === slot.player_id)} /></div><SearchSelect label={`${ROLE_LABEL[slot.role]} · ${side === 'blue' ? 'xanh' : 'đỏ'}`} options={playerOptions} value={slot.player_id} exclude={playerIds.filter(id => id !== slot.player_id)} disabled={contextLocked} placeholder="Chọn tuyển thủ…" onChange={id => setContext(previous => ({ ...previous, [`${side}_roster`]: previous[`${side}_roster`].map((row, i) => i === index ? { ...row, player_id: id } : row), user_confirmed: false }))} /></div>)}</div>
              {team && data.rosters[team.id] && <p className="roster-note">Gợi ý từ đội hình ngày {date(data.rosters[team.id].source_ended_at)}. Hãy kiểm tra lại tuyển thủ và vị trí.</p>}
            </div>
          })}</div>
          <div className="context-action"><label className="confirmation"><input type="checkbox" checked={context.user_confirmed} disabled={contextLocked} onChange={event => setContext(value => ({ ...value, user_confirmed: event.target.checked }))} /><span>Tôi đã kiểm tra hai đội, bên thi đấu, patch và mười tuyển thủ theo vị trí.</span></label>
            {!pre && <button className="primary" disabled={!!busy || !contextComplete} onClick={() => void createPre()}>{preRequest.current ? 'Thử lưu PRE lại →' : 'Tạo đánh giá PRE →'}</button>}
            {pre && <span className="saved-label">✓ Đã lưu PRE #{pre.pre_evaluation_id}</span>}
          </div></section>
          {busy && <div className="busy-banner" role="status"><span className="spinner" />{busy}</div>}
          {error && <div className="error-banner" role="alert">{error}</div>}
          <div ref={resultAnchor} className="result-anchor">{result && <><Comparison result={result} selectedFamily={data.metrics.selected_family} /><Evidence result={result} /></>}</div>
          {pre && <section className="panel draft-panel" aria-label="Đội hình tướng"><div className="panel-heading"><div><span className="eyebrow">02 / SAU CẤM CHỌN</span><h2>Đội hình cuối cùng</h2><p>Gắn đúng tướng với tuyển thủ và vị trí. PRE #{pre.pre_evaluation_id} được giữ làm mốc nền.</p></div>{post || postRequest.current ? <button className="secondary" disabled={!!busy} onClick={() => void edit(true)}>Chỉnh sửa đội hình</button> : <span className="tag">{champions.filter(Boolean).length} / 10 tướng</span>}</div>
            <div className="draft-grid">{(['blue', 'red'] as Side[]).map((side, sideIndex) => <div key={side} className={`draft-side ${side}`}><h3><span className={`side-dot ${side}`} />{pre.teams[side].name}</h3>{pre.roster?.[side].map((slot, i) => {
              const index = sideIndex * 5 + i
              const champion = data.champions.find(row => row.id === champions[index])
              return <div className="champion-slot" key={slot.role}><Avatar entity={champion} /><div className="champion-input"><span className="player-name">{slot.player.name}</span><SearchSelect label={`${ROLE_LABEL[slot.role]} · tướng ${side === 'blue' ? 'xanh' : 'đỏ'}`} options={data.champions} value={champions[index]} disabled={lineupLocked} exclude={champions.filter((_, j) => j !== index)} placeholder="Chọn tướng…" onChange={id => setChampions(previous => previous.map((value, j) => j === index ? id : value))} /></div></div>
            })}</div>)}</div>
            <div className="draft-action"><p>{post ? `Đã lưu POST #${post.post_evaluation_id}. Kết quả so sánh nằm phía trên.` : 'Tạo POST sau khi xác nhận đủ mười tướng, trước khi ván bắt đầu.'}</p>{!post && <button className="primary" disabled={!lineupComplete || !!busy} onClick={() => void createPost()}>{postRequest.current ? 'Thử lưu POST lại →' : 'Tạo POST & so sánh →'}</button>}</div>
          </section>}
          <Quality data={data} />
          <div className="source-note"><span className="info-symbol">i</span><p>Ước lượng từ dữ liệu lịch sử chuyên nghiệp, mới nhất trong kho: {date(data.history_latest)}. Bối cảnh và đội hình do bạn xác nhận; thông tin chỉ dùng để tham khảo trước trận.</p></div>
        </>}
      </>}
    </main>
    <footer><span className="footer-brand">MATCH INSIGHT</span><span>PRE / POST · Lịch sử chuyên nghiệp · Ba mô hình đối chiếu</span></footer>
  </div>
}
