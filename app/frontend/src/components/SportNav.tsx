/** Links between the karate page (/) and the swimming page (/swim). */
export function SportNav({ active }: { active: 'karate' | 'swimming' }) {
  return (
    <nav className="sports">
      <a href="/" className={active === 'karate' ? 'active' : ''}>
        🥋 Karate
      </a>
      <a href="/swim" className={active === 'swimming' ? 'active' : ''}>
        🏊 Swimming
      </a>
    </nav>
  )
}
