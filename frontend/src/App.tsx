import { useState } from 'react'
import './App.css'

export default function App() {
  const [photo, setPhoto] = useState<File | null>(null)
  const [photoUrl, setPhotoUrl] = useState('')
  const [resultUrl, setResultUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setPhoto(file)
    setPhotoUrl(URL.createObjectURL(file))
    setResultUrl('')
    setError('')
  }

  async function generate() {
    if (!photo || loading) return
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('photo', photo)
      const res = await fetch('/api/generate', { method: 'POST', body: form })
      if (!res.ok) {
        const detail = await res
          .json()
          .then((b) => b.detail)
          .catch(() => res.statusText)
        throw new Error(String(detail))
      }
      setResultUrl(URL.createObjectURL(await res.blob()))
    } catch (e) {
      setError(e instanceof Error ? e.message : '생성에 실패했습니다.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main>
      <h1>MonSnap</h1>
      <p className="tagline">사진을 찍으면 나만의 몬스터가 태어납니다</p>

      <label className="pick">
        {/* capture 미지정: 모바일에서 카메라/앨범 선택 시트가 뜬다 */}
        <input type="file" accept="image/*" onChange={onPick} hidden />
        📷 사진 찍기 / 선택
      </label>

      {photoUrl && (
        <div className="panes">
          <figure>
            <img src={photoUrl} alt="원본 사진" />
            <figcaption>원본</figcaption>
          </figure>
          <figure>
            {resultUrl ? (
              <img src={resultUrl} alt="생성된 몬스터" />
            ) : (
              <div className="placeholder">
                {loading ? '몬스터 소환 중… (최대 2분)' : '?'}
              </div>
            )}
            <figcaption>몬스터</figcaption>
          </figure>
        </div>
      )}

      {photo && (
        <button onClick={generate} disabled={loading}>
          {loading ? '생성 중…' : resultUrl ? '다시 생성' : '몬스터 생성'}
        </button>
      )}
      {resultUrl && (
        <a className="download" href={resultUrl} download="monster.png">
          이미지 저장
        </a>
      )}
      {error && <p className="error">{error}</p>}
    </main>
  )
}
