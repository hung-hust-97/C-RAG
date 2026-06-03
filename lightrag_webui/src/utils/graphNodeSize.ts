import * as Constants from '../lib/constants'

/**
 * Map entity_type (from KG) to legal hierarchy level.
 *
 * Level 0: LEGAL_DOCUMENT (tvpl_key — root document)
 * Level 1: CHAPTER (Chương)
 * Level 2: ARTICLE (Điều)
 * Level 3: CLAUSE (Khoản)
 * Level 4: POINT (Điểm)
 *
 * Returns null for non-legal entity types (domain entities extracted by LLM).
 */
const LEGAL_ENTITY_TYPE_TO_LEVEL: Record<string, number> = {
  'legal_document': 0,
  'chapter': 1,
  'article': 2,
  'clause': 3,
  'point': 4,
}

/**
 * Sizes for each legal hierarchy level (0-5), decreasing evenly.
 * Level 5 is for domain entities (LLM-extracted) in a legal graph.
 */
const LEGAL_SIZES_BY_LEVEL = [24, 20, 16, 12, 9, 6]

export const getLegalNodeLevel = (properties: Record<string, any> | undefined): number | null => {
  if (!properties) return null
  const entityType = String(properties.entity_type || '').toLowerCase()
  const level = LEGAL_ENTITY_TYPE_TO_LEVEL[entityType]
  return level !== undefined ? level : null
}

export const getLegalNodeSize = (level: number): number => {
  return LEGAL_SIZES_BY_LEVEL[level] ?? Constants.minNodeSize
}

export const getDegreeNodeSize = (degree: number, minDegree: number, maxDegree: number): number => {
  const range = maxDegree - minDegree
  if (range <= 0) {
    return 10
  }

  const scale = Constants.maxNodeSize - Constants.minNodeSize
  return Math.round(
    Constants.minNodeSize + scale * Math.pow((degree - minDegree) / range, 0.5)
  )
}


/**
 * Get display size for a node.
 *
 * - Legal hierarchy nodes (LEGAL_DOCUMENT/CHAPTER/ARTICLE/CLAUSE/POINT) → size by level
 * - In a legal graph, domain entities (non-hierarchy) → level 5 size (smallest)
 * - In a general graph → size by degree
 */
export const getNodeDisplaySize = (
  node: { degree: number; properties: Record<string, any> },
  minDegree: number,
  maxDegree: number,
  degreeOverride?: number,
  graphIsLegal?: boolean
): number => {
  const legalLevel = getLegalNodeLevel(node.properties)
  if (legalLevel !== null) {
    return getLegalNodeSize(legalLevel)
  }
  // In a legal graph, entities without hierarchy level get smallest size (level 5)
  if (graphIsLegal) {
    return LEGAL_SIZES_BY_LEVEL[5]
  }
  return getDegreeNodeSize(degreeOverride ?? node.degree, minDegree, maxDegree)
}
