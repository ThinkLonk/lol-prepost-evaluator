import type { Entity } from './types'

const words = (value: string) => value.toLocaleLowerCase('vi').normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').match(/[\p{L}\p{N}]+/gu) ?? []

export function searchEntities(options: Entity[], query: string, exclude: string[] = []): Entity[] {
  const term = words(query).join('')
  const excluded = new Set(exclude)
  return options.filter(option => !excluded.has(option.id)).map(option => {
    const tokens = words(option.name)
    const name = tokens.join('')
    const initials = tokens.map(token => token[0]).join('')
    const rank = !term ? 0 : name === term || initials === term ? 0
      : name.startsWith(term) ? 1
      : term.length >= 2 && initials.startsWith(term) ? 2
      : name.includes(term) ? 3 : -1
    return { option, rank, length: name.length }
  }).filter(item => item.rank >= 0)
    .sort((a, b) => a.rank - b.rank || (term ? a.length - b.length : 0))
    .slice(0, 40).map(item => item.option)
}
