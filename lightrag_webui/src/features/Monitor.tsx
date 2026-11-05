import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useSettingsStore } from '@/stores/settings'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import Checkbox from '@/components/ui/Checkbox'
import PaginationControls from '@/components/ui/PaginationControls'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose
} from '@/components/ui/Dialog'
import Badge from '@/components/ui/Badge'
import {
  getPipelineStageRuns,
  getAllWorkspacesPipelineStageRuns,
  reprocessFailedDocuments,
  approveStage,
  replayFailedStage,
  bulkApproveStage,
  getStageSpeed,
  getPipelineConfig,
  updatePipelineConfig,
  PipelineStageRunResponse,
  PipelineDocType,
  StageSpeedEntry,
  getWorkspaces,
  type WorkspaceItem
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { toast } from 'sonner'
import {
  RefreshCwIcon,
  RotateCcwIcon,
  PlayIcon,
  AlertTriangleIcon,
  SearchIcon,
  XIcon,
  CopyIcon,
  CheckIcon,
  FileTextIcon,
  FastForwardIcon,
  ZapIcon,
  SettingsIcon,
  SaveIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  PlusIcon,
  Trash2Icon,
  ScaleIcon,
  LayersIcon
} from 'lucide-react'

const CORE_STAGES = [
  { id: 'extraction', label: 'Extraction' },
  { id: 'chunking', label: 'Chunking' },
  { id: 'chunk_embedding', label: 'Chunk Embed' },
  { id: 'entity_extraction', label: 'Entity Extract' },
  { id: 'kg_merge', label: 'KG Merge' },
  { id: 'kg_embedding', label: 'KG Embed' },
  { id: 'finalize', label: 'Finalize' }
]

const STATUS_OPTIONS = [
  { id: 'running', label: 'Running' },
  { id: 'paused', label: 'Paused' },
  { id: 'done', label: 'Done' },
  { id: 'failed', label: 'Failed' }
]

const REALTIME_REFRESH_INTERVAL_MS = 2000
const SPEED_REFRESH_INTERVAL_MS = 10000

const compareWorkspaceByName = (a: WorkspaceItem, b: WorkspaceItem) => {
  const aName = (a.workspace_name || a.workspace_id).trim()
  const bName = (b.workspace_name || b.workspace_id).trim()
  return (
    aName.localeCompare(bName, undefined, { sensitivity: 'base', numeric: true }) ||
    a.workspace_id.localeCompare(b.workspace_id, undefined, { sensitivity: 'base', numeric: true })
  )
}

export default function Monitor() {
  const { t } = useTranslation()
  const selectedWorkspaceId = useSettingsStore.use.selectedWorkspaceId()
  // Workspace selection state
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([])
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string>('all')

  // Table state
  const [runs, setRuns] = useState<PipelineStageRunResponse[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [loading, setLoading] = useState(false)
  const [statusMatrix, setStatusMatrix] = useState<Record<string, Record<string, number>>>({})
  const [stageSpeed, setStageSpeed] = useState<Record<string, StageSpeedEntry>>({})
  const [realtimeEnabled, setRealtimeEnabled] = useState(true)
  const [pageVisible, setPageVisible] = useState(() => document.visibilityState === 'visible')
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null)
  const fetchInFlightRef = useRef(false)
  const lastSpeedFetchRef = useRef(0)

  // Fetch workspaces list
  useEffect(() => {
    const loadWorkspaces = async () => {
      try {
        const response = await getWorkspaces()
        const apiWorkspaces = (response.workspace_items || [])
          .map((item) => ({
            workspace_id: item.workspace_id?.trim() || '',
            workspace_name: item.workspace_name?.trim() || item.workspace_id?.trim() || ''
          }))
          .filter((item) => !!item.workspace_id)
          .sort(compareWorkspaceByName)
        setWorkspaces(apiWorkspaces)
      } catch (error) {
        console.error('Failed to load workspaces:', error)
      }
    }
    loadWorkspaces()
  }, [])

  // Keep a manual "All Workspaces" monitor selection from being overwritten by
  // the global workspace selector after approve/refresh side effects.
  useEffect(() => {
    setActiveWorkspaceId((current) => {
      if (current === 'all') return current
      return selectedWorkspaceId && selectedWorkspaceId !== 'default'
        ? selectedWorkspaceId
        : 'all'
    })
  }, [selectedWorkspaceId])

  // Filter state
  const [search, setSearch] = useState('')
  const [searchDebounced, setSearchDebounced] = useState('')
  const [selectedStage, setSelectedStage] = useState<string | null>(null)
  const [selectedStatuses, setSelectedStatuses] = useState<string[]>([])
  const [activeDocType, setActiveDocType] = useState<PipelineDocType>('all')
  const [docTypeCounts, setDocTypeCounts] = useState<{ all: number; legal: number; general: number }>({
    all: 0,
    legal: 0,
    general: 0
  })

  // Selection state
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([])

  // Error modal state
  const [errorModalOpen, setErrorModalOpen] = useState(false)
  const [activeError, setActiveError] = useState<{
    docId: string
    documentName: string | null
    stage: string
    shard: string
    error: any
  } | null>(null)
  const [copied, setCopied] = useState(false)

  type ConfigNode = {
    host: string
    model: string
    apiKey: string
  }

  // Config panel state
  const [configOpen, setConfigOpen] = useState(false)
  const [configLoading, setConfigLoading] = useState(false)
  const [configSaving, setConfigSaving] = useState(false)
  const [configNodes, setConfigNodes] = useState<ConfigNode[]>([{ host: '', model: '', apiKey: '' }])
  const [configPausedStages, setConfigPausedStages] = useState<string[]>([])

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearchDebounced(search)
      setPage(1) // Reset page on search change
    }, 400)
    return () => clearTimeout(timer)
  }, [search])

  // Reset state when activeWorkspaceId or activeDocType changes
  useEffect(() => {
    setPage(1)
    setSelectedDocIds([])
  }, [activeWorkspaceId, activeDocType])

  // Fetch runs
  const fetchRuns = useCallback(async (options?: { showLoading?: boolean; quiet?: boolean }) => {
    if (fetchInFlightRef.current) return

    const showLoading = options?.showLoading ?? true
    const quiet = options?.quiet ?? false
    fetchInFlightRef.current = true
    if (showLoading) setLoading(true)
    try {
      const params = {
        workspace_id: activeWorkspaceId,
        stage: selectedStage,
        status: selectedStatuses.length > 0 ? selectedStatuses : null,
        search: searchDebounced || null,
        doc_type: activeDocType !== 'all' ? activeDocType : null,
        page,
        page_size: pageSize
      }
      const response = activeWorkspaceId === 'all'
        ? await getAllWorkspacesPipelineStageRuns(params)
        : await getPipelineStageRuns(params)
      setRuns(response.runs)
      setTotal(response.total)
      setStatusMatrix(response.status_matrix)
      if (response.doc_type_counts) {
        setDocTypeCounts(response.doc_type_counts)
      }
      setLastUpdatedAt(new Date())

      // Fetch speed stats alongside if not querying all workspaces
      if (activeWorkspaceId !== 'all') {
        const now = Date.now()
        if (now - lastSpeedFetchRef.current > SPEED_REFRESH_INTERVAL_MS || showLoading) {
          try {
            const speedData = await getStageSpeed()
            const speedMap: Record<string, StageSpeedEntry> = {}
            for (const entry of speedData) {
              speedMap[entry.stage] = entry
            }
            setStageSpeed(speedMap)
            lastSpeedFetchRef.current = now
          } catch { /* speed is optional */ }
        }
      } else {
        setStageSpeed({})
      }
    } catch (error) {
      console.error('Failed to load stage runs:', error)
      if (!quiet) {
        toast.error(t('monitorPanel.errors.loadFailed', 'Failed to load pipeline runs'))
      }
    } finally {
      fetchInFlightRef.current = false
      if (showLoading) setLoading(false)
    }
  }, [activeWorkspaceId, selectedStage, selectedStatuses, searchDebounced, activeDocType, page, pageSize, t])

  // Fetch on change
  useEffect(() => {
    fetchRuns({ showLoading: true })
  }, [fetchRuns])

  useEffect(() => {
    const handleVisibilityChange = () => {
      setPageVisible(document.visibilityState === 'visible')
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [])

  useEffect(() => {
    if (!realtimeEnabled || !pageVisible) return

    const intervalId = window.setInterval(() => {
      fetchRuns({ showLoading: false, quiet: true })
    }, REALTIME_REFRESH_INTERVAL_MS)

    return () => window.clearInterval(intervalId)
  }, [fetchRuns, realtimeEnabled, pageVisible])

  // Refresh helper
  const handleRefresh = () => {
    fetchRuns({ showLoading: true })
    setSelectedDocIds([])
    toast.success(t('common.refreshed', 'Refreshed'))
  }

  const realtimeStatusText = !realtimeEnabled
    ? t('monitorPanel.realtimeOff', 'Live off')
    : pageVisible
      ? t('monitorPanel.realtimeOn', 'Live')
      : t('monitorPanel.realtimePaused', 'Paused')

  const lastUpdatedText = lastUpdatedAt
    ? lastUpdatedAt.toLocaleTimeString()
    : '-'

  // Format started/completed duration
  const getDurationText = (start: string | null, end: string | null) => {
    if (!start) return '-'
    const startTime = new Date(start).getTime()
    const endTime = end ? new Date(end).getTime() : Date.now()
    const diffMs = endTime - startTime
    if (diffMs < 0) return '0s'
    const diffSecs = Math.round(diffMs / 1000)
    if (diffSecs < 60) return `${diffSecs}s`
    const diffMins = Math.floor(diffSecs / 60)
    const remainingSecs = diffSecs % 60
    return `${diffMins}m ${remainingSecs}s`
  }

  // Card aggregation helper
  const getStageCount = (stageId: string, status: string): number => {
    return statusMatrix[stageId]?.[status] || 0
  }

  // Filter click helper
  const handleStageStatusClick = (stageId: string, status: string) => {
    setSelectedStage(stageId)
    setSelectedStatuses([status])
    setPage(1)
  }

  const resetFilters = () => {
    setSelectedStage(null)
    setSelectedStatuses([])
    setSearch('')
    setActiveDocType('all')
    setPage(1)
  }

  // Checkbox handlers
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const pageDocIds = runs.map((r) => r.doc_id).filter(Boolean)
      setSelectedDocIds((prev) => Array.from(new Set([...prev, ...pageDocIds])))
    } else {
      const pageDocIds = runs.map((r) => r.doc_id)
      setSelectedDocIds((prev) => prev.filter((id) => !pageDocIds.includes(id)))
    }
  }

  const handleSelectRow = (docId: string, checked: boolean) => {
    if (checked) {
      setSelectedDocIds((prev) => [...prev, docId])
    } else {
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId))
    }
  }

  const isAllPageSelected = useMemo(() => {
    if (runs.length === 0) return false
    return runs.every((r) => selectedDocIds.includes(r.doc_id))
  }, [runs, selectedDocIds])

  const isSomePageSelected = useMemo(() => {
    if (runs.length === 0) return false
    return runs.some((r) => selectedDocIds.includes(r.doc_id)) && !isAllPageSelected
  }, [runs, selectedDocIds, isAllPageSelected])

  // Reprocess handlers
  const handleReprocessSelected = async () => {
    if (selectedDocIds.length === 0) return
    const promise = reprocessFailedDocuments(selectedDocIds)
    toast.promise(promise, {
      loading: t('monitorPanel.reprocessing', 'Initiating reprocessing...'),
      success: () => {
        setSelectedDocIds([])
        setTimeout(fetchRuns, 1500) // Delay to let workers pick it up
        return t('monitorPanel.reprocessSuccess', 'Reprocessing initiated successfully')
      },
      error: (err) => `${t('monitorPanel.reprocessFailed', 'Failed to reprocess')}: ${errorMessage(err)}`
    })
  }

  const handleReprocessAllFailed = async () => {
    const promise = reprocessFailedDocuments(undefined, activeDocType)
    const toastLoading = activeDocType === 'legal'
      ? t('monitorPanel.reprocessingLegal', 'Initiating reprocessing for legal documents...')
      : activeDocType === 'general'
        ? t('monitorPanel.reprocessingGeneral', 'Initiating reprocessing for general documents...')
        : t('monitorPanel.reprocessing', 'Initiating reprocessing...')

    toast.promise(promise, {
      loading: toastLoading,
      success: (result) => {
        setTimeout(fetchRuns, 1500)
        return result?.message || t('monitorPanel.reprocessSuccess', 'Reprocessing initiated successfully')
      },
      error: (err) => `${t('monitorPanel.reprocessFailed', 'Failed to reprocess')}: ${errorMessage(err)}`
    })
  }

  // Approve stage handler
  const handleApproveStage = async (runId: string, stage: string, shardIndex: number, workspaceId?: string) => {
    const promise = approveStage(runId, stage, shardIndex, workspaceId)
    toast.promise(promise, {
      loading: t('monitorPanel.approving', 'Approving stage...'),
      success: () => {
        setTimeout(fetchRuns, 1500)
        return t('monitorPanel.approveSuccess', 'Stage approved successfully')
      },
      error: (err) => `Failed to approve stage: ${errorMessage(err)}`
    })
  }

  const handleReplayStage = async (
    run: PipelineStageRunResponse
  ) => {
    const reason = window.prompt(
      t('monitorPanel.replayReasonPrompt', 'Why should this failed stage be replayed?')
    )
    if (!reason?.trim()) return

    const promise = replayFailedStage(
      run.run_id,
      run.stage,
      run.shard_index,
      run.workspace_id,
      reason.trim()
    )
    toast.promise(promise, {
      loading: t('monitorPanel.replayingStage', 'Queueing stage replay...'),
      success: (result) => {
        setTimeout(fetchRuns, 1500)
        return result.message
      },
      error: (err) => `${t('monitorPanel.replayFailed', 'Failed to queue stage replay')}: ${errorMessage(err)}`
    })
  }

  const handleReprocessSingle = async (docId: string) => {
    const promise = reprocessFailedDocuments([docId])
    toast.promise(promise, {
      loading: t('monitorPanel.reprocessing', 'Initiating reprocessing...'),
      success: () => {
        setTimeout(fetchRuns, 1500)
        return t('monitorPanel.reprocessSuccess', 'Reprocessing initiated successfully')
      },
      error: (err) => `${t('monitorPanel.reprocessFailed', 'Failed to reprocess')}: ${errorMessage(err)}`
    })
  }

  // Bulk approve a stage
  const handleBulkApproveStage = async (stageId: string) => {
    const promise = bulkApproveStage(stageId, activeWorkspaceId)
    toast.promise(promise, {
      loading: `Approving all paused for ${stageId}...`,
      success: (data) => {
        setTimeout(fetchRuns, 1500)
        return data.message || 'Stage approved'
      },
      error: (err) => `Failed to approve: ${errorMessage(err)}`
    })
  }

  // Copy error trace
  const copyTrace = () => {
    if (!activeError) return
    const text = activeError.error?.traceback || JSON.stringify(activeError.error, null, 2)
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
    toast.success(t('retrievePanel.chatMessage.copySuccess', 'Copied to clipboard'))
  }

  return (
    <div className="flex h-full w-full flex-col gap-6 p-6 overflow-y-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            {t('monitorPanel.title', 'Pipeline Stage Monitor')}
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
            {t('monitorPanel.description', 'Monitor and reprocess document processing steps.')}
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <div className="flex items-center gap-2 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-background px-2.5 py-1.5">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setRealtimeEnabled((value) => !value)}
              className="h-7 px-2 text-xs"
            >
              <span
                className={cn(
                  'mr-1.5 size-2 rounded-full',
                  realtimeEnabled && pageVisible
                    ? 'bg-emerald-500 animate-pulse'
                    : 'bg-zinc-400'
                )}
              />
              {realtimeStatusText}
            </Button>
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-mono">
              {lastUpdatedText}
            </span>
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
            <RefreshCwIcon className={cn('size-4 mr-2', loading && 'animate-spin')} />
            {t('documentPanel.documentManager.refreshTooltip', 'Refresh')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReprocessAllFailed}
            className="border-red-200 dark:border-red-900/50 hover:bg-red-50 dark:hover:bg-red-950/20 text-red-600 dark:text-red-400"
          >
            <AlertTriangleIcon className="size-4 mr-2" />
            {activeDocType === 'legal'
              ? t('monitorPanel.reprocessAllFailedLegal', 'Reprocess Failed Legal')
              : activeDocType === 'general'
                ? t('monitorPanel.reprocessAllFailedGeneral', 'Reprocess Failed General')
                : t('monitorPanel.reprocessAllFailed', 'Reprocess All Failed')}
          </Button>
        </div>
      </div>

      {/* Pipeline Configuration Panel */}
      <div className="border rounded-xl bg-background overflow-visible">
        <button
          onClick={async () => {
            const opening = !configOpen
            setConfigOpen(opening)

            if (opening) {
              setConfigLoading(true)
              try {
                const config = await getPipelineConfig()

                const hosts = (config.entity_extraction_host || '')
                  .split(',')
                  .map((s) => s.trim())

                const models = (config.entity_extraction_model || '')
                  .split(',')
                  .map((s) => s.trim())

                const apiKeys = (config.entity_extraction_api_key || '')
                  .split(',')
                  .map((s) => s.trim())

                const parsedNodes: ConfigNode[] = []
                const nodeCount = Math.max(hosts.length, models.length, apiKeys.length)

                for (let i = 0; i < nodeCount; i++) {
                  if (hosts[i] || models[i] || apiKeys[i]) {
                    parsedNodes.push({
                      host: hosts[i] || '',
                      model: models[i] || '',
                      apiKey: apiKeys[i] || ''
                    })
                  }
                }

                if (parsedNodes.length === 0) {
                  parsedNodes.push({ host: '', model: '', apiKey: '' })
                }

                setConfigNodes(parsedNodes)

                setConfigPausedStages(
                  config.paused_stages
                    ? config.paused_stages
                      .split(',')
                      .map((s) => s.trim())
                      .filter(Boolean)
                    : []
                )
              } catch {
                // ignore
              } finally {
                setConfigLoading(false)
              }
            }
          }}
          className="w-full flex items-center justify-between p-3 hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors rounded-t-xl"
        >
          <div className="flex items-center gap-2 min-w-0">
            <SettingsIcon className="size-4 text-zinc-500 shrink-0" />
            <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 truncate">
              Pipeline Configuration
            </span>
          </div>

          {configOpen ? (
            <ChevronUpIcon className="size-4 text-zinc-400 shrink-0" />
          ) : (
            <ChevronDownIcon className="size-4 text-zinc-400 shrink-0" />
          )}
        </button>

        {configOpen && (
          <div className="border-t p-4 space-y-4 max-h-[70vh] overflow-y-auto overflow-x-hidden">
            {configLoading ? (
              <div className="text-sm text-zinc-400 text-center py-4">
                Loading config...
              </div>
            ) : (
              <>
                {/* Entity Extraction Settings */}
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h3 className="text-xs font-bold uppercase text-zinc-500 tracking-wider">
                      Entity Extraction LLM Nodes
                    </h3>

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setConfigNodes((prev) => [
                          ...prev,
                          { host: '', model: '', apiKey: '' }
                        ])
                      }
                      className="h-7 w-fit text-xs px-2 border-dashed hover:border-emerald-500 hover:text-emerald-500 transition-colors"
                    >
                      <PlusIcon className="size-3.5 mr-1" />
                      Add Node
                    </Button>
                  </div>

                  <div className="space-y-3">
                    {configNodes.map((node, index) => (
                      <div
                        key={index}
                        className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50/40 dark:bg-zinc-900/30 space-y-3 relative"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[11px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
                            Node {index + 1}
                          </span>

                          {configNodes.length > 1 && (
                            <button
                              onClick={() =>
                                setConfigNodes((prev) =>
                                  prev.filter((_, i) => i !== index)
                                )
                              }
                              className="text-zinc-400 hover:text-red-500 transition-colors shrink-0"
                              title="Remove Node"
                            >
                              <Trash2Icon className="size-3.5" />
                            </button>
                          )}
                        </div>

                        <div className="space-y-2.5">
                          <div className="grid grid-cols-1 sm:grid-cols-[96px_1fr] gap-1.5 sm:gap-2 sm:items-center">
                            <label className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400">
                              Host
                            </label>

                            <input
                              type="text"
                              value={node.host}
                              onChange={(e) => {
                                const newNodes = [...configNodes]
                                newNodes[index] = { ...node, host: e.target.value }
                                setConfigNodes(newNodes)
                              }}
                              className="w-full min-w-0 h-8 px-2.5 rounded-md border border-zinc-200 dark:border-zinc-800 bg-background text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
                              placeholder="e.g. http://10.1.6.52:8004/v1"
                            />
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-[96px_1fr] gap-1.5 sm:gap-2 sm:items-center">
                            <label className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400">
                              Model
                            </label>

                            <input
                              type="text"
                              value={node.model}
                              onChange={(e) => {
                                const newNodes = [...configNodes]
                                newNodes[index] = { ...node, model: e.target.value }
                                setConfigNodes(newNodes)
                              }}
                              className="w-full min-w-0 h-8 px-2.5 rounded-md border border-zinc-200 dark:border-zinc-800 bg-background text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
                              placeholder="e.g. google/gemma-4-26b-a4b-it"
                            />
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-[96px_1fr] gap-1.5 sm:gap-2 sm:items-center">
                            <label className="text-[10px] font-semibold text-zinc-500 dark:text-zinc-400">
                              API Key
                            </label>

                            <input
                              type="password"
                              value={node.apiKey}
                              onChange={(e) => {
                                const newNodes = [...configNodes]
                                newNodes[index] = { ...node, apiKey: e.target.value }
                                setConfigNodes(newNodes)
                              }}
                              className="w-full min-w-0 h-8 px-2.5 rounded-md border border-zinc-200 dark:border-zinc-800 bg-background text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500"
                              placeholder="Optional api key"
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Paused Stages */}
                <div className="space-y-3">
                  <h3 className="text-xs font-bold uppercase text-zinc-500 tracking-wider">Paused Stages</h3>
                  <div className="flex flex-wrap gap-2">
                    {CORE_STAGES.map((stage) => {
                      const isChecked = configPausedStages.includes(stage.id)
                      return (
                        <button
                          key={stage.id}
                          onClick={() =>
                            setConfigPausedStages((prev) =>
                              isChecked ? prev.filter((s) => s !== stage.id) : [...prev, stage.id]
                            )
                          }
                          className={cn(
                            'h-8 px-3 rounded-lg border text-xs font-medium transition-all duration-150',
                            isChecked
                              ? 'bg-amber-50 dark:bg-amber-950/30 border-amber-300 dark:border-amber-800 text-amber-700 dark:text-amber-400'
                              : 'border-zinc-200 dark:border-zinc-800 bg-background text-zinc-500 hover:bg-zinc-50 dark:hover:bg-zinc-900'
                          )}
                        >
                          {stage.label}
                        </button>
                      )
                    })}
                  </div>
                  <p className="text-[10px] text-zinc-400">
                    Paused stages will block documents from proceeding. Workers auto-restart on save.
                  </p>
                </div>

                {/* Save */}
                <div className="flex justify-end">
                  <Button
                    variant="default"
                    size="sm"
                    disabled={configSaving}
                    onClick={async () => {
                      setConfigSaving(true)

                      const hosts = configNodes.map((n) => n.host.trim()).filter(Boolean).join(',')
                      const models = configNodes.map((n) => n.model.trim()).filter(Boolean).join(',')
                      const apiKeys = configNodes.map((n) => n.apiKey.trim()).join(',')
                      const serializedApiKeys = configNodes.some((n) => n.apiKey.trim()) ? apiKeys : ''

                      const promise = updatePipelineConfig({
                        entity_extraction_model: models,
                        entity_extraction_host: hosts,
                        entity_extraction_api_key: serializedApiKeys,
                        paused_stages: configPausedStages.join(','),
                        restart_workers: true,
                      })
                      toast.promise(promise, {
                        loading: 'Saving config & restarting workers...',
                        success: (d) => {
                          setConfigSaving(false)
                          return d.message || 'Config saved'
                        },
                        error: (err) => {
                          setConfigSaving(false)
                          return `Failed: ${errorMessage(err)}`
                        },
                      })
                    }}
                  >
                    <SaveIcon className="size-4 mr-2" />
                    Save & Restart Workers
                  </Button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Document Type Tabs Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-zinc-50/80 dark:bg-zinc-900/40 p-2.5 rounded-xl border border-zinc-200/80 dark:border-zinc-800">
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Tab 1: All */}
          <button
            onClick={() => {
              setActiveDocType('all')
              setPage(1)
            }}
            className={cn(
              'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all duration-150',
              activeDocType === 'all'
                ? 'bg-emerald-500 text-white shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800/60'
            )}
          >
            <LayersIcon className="size-3.5" />
            <span>{t('monitorPanel.tabs.all', 'All Documents')}</span>
            <span
              className={cn(
                'px-1.5 py-0.5 rounded-full text-[10px] font-mono',
                activeDocType === 'all'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300'
              )}
            >
              {docTypeCounts.all}
            </span>
          </button>

          {/* Tab 2: Legal Documents */}
          <button
            onClick={() => {
              setActiveDocType('legal')
              setPage(1)
            }}
            className={cn(
              'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all duration-150',
              activeDocType === 'legal'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800/60'
            )}
          >
            <ScaleIcon className="size-3.5" />
            <span>{t('monitorPanel.tabs.legal', 'Legal Documents')}</span>
            <span
              className={cn(
                'px-1.5 py-0.5 rounded-full text-[10px] font-mono',
                activeDocType === 'legal'
                  ? 'bg-purple-700 text-white'
                  : 'bg-purple-100 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300'
              )}
            >
              {docTypeCounts.legal}
            </span>
          </button>

          {/* Tab 3: General Documents */}
          <button
            onClick={() => {
              setActiveDocType('general')
              setPage(1)
            }}
            className={cn(
              'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all duration-150',
              activeDocType === 'general'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800/60'
            )}
          >
            <FileTextIcon className="size-3.5" />
            <span>{t('monitorPanel.tabs.general', 'General Documents')}</span>
            <span
              className={cn(
                'px-1.5 py-0.5 rounded-full text-[10px] font-mono',
                activeDocType === 'general'
                  ? 'bg-blue-700 text-white'
                  : 'bg-blue-100 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300'
              )}
            >
              {docTypeCounts.general}
            </span>
          </button>
        </div>

        <div className="text-xs text-zinc-500 dark:text-zinc-400 font-medium px-1">
          {activeDocType === 'legal' && (
            <span className="text-purple-600 dark:text-purple-400">
              ⚖️ {t('monitorPanel.filterActiveLegal', 'Viewing Vietnamese Legal Documents')}
            </span>
          )}
          {activeDocType === 'general' && (
            <span className="text-blue-600 dark:text-blue-400">
              📄 {t('monitorPanel.filterActiveGeneral', 'Viewing General Documents')}
            </span>
          )}
          {activeDocType === 'all' && (
            <span>{t('monitorPanel.filterActiveAll', 'Viewing all document types')}</span>
          )}
        </div>
      </div>

      {/* Stage Cards Matrix */}
      <div>
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 mb-3">
          {t('monitorPanel.stageCountsTitle', 'Stage Progress Summary')}
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7 gap-3">
          {CORE_STAGES.map((stage, stageIdx) => {
            const runningCount = getStageCount(stage.id, 'running')
            const pausedCount = getStageCount(stage.id, 'paused')
            const doneCount = getStageCount(stage.id, 'done')
            const failedCount = getStageCount(stage.id, 'failed')
            const totalCount = runningCount + pausedCount + doneCount + failedCount

            const isFilteredThisStage = selectedStage === stage.id
            const speed = stageSpeed[stage.id]
            const docsPerMin = speed ? Math.round(speed.docs_completed_5m / 5) : 0
            const avgDur = speed?.avg_duration_seconds

            return (
              <Card
                key={stage.id}
                className={cn(
                  'border-t-4 transition-all duration-200 relative overflow-hidden',
                  isFilteredThisStage
                    ? 'border-t-emerald-400 ring-2 ring-emerald-500/20 shadow-md bg-emerald-50/5 dark:bg-emerald-950/5'
                    : 'border-t-zinc-300 dark:border-t-zinc-700 hover:shadow-sm'
                )}
              >
                <CardHeader className="p-3 pb-1">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-xs font-bold truncate text-zinc-800 dark:text-zinc-100">
                      <span className="text-[10px] text-zinc-400 mr-1">{stageIdx + 1}.</span>
                      {stage.label}
                    </CardTitle>
                    {pausedCount > 0 && (
                      <button
                        onClick={() => handleBulkApproveStage(stage.id)}
                        className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 hover:bg-amber-200 dark:hover:bg-amber-900/50 font-semibold transition-colors flex items-center gap-1"
                        title={`Approve all ${pausedCount} paused`}
                      >
                        <FastForwardIcon className="size-3" />
                        Approve
                      </button>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="p-3 pt-0 flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-[10px] text-zinc-500 border-b pb-1">
                    <span>Total</span>
                    <span className="font-semibold text-zinc-700 dark:text-zinc-300">{totalCount}</span>
                  </div>

                  <div className="grid grid-cols-4 gap-0.5 text-center">
                    {/* Running */}
                    <button
                      onClick={() => handleStageStatusClick(stage.id, 'running')}
                      className={cn(
                        'flex flex-col p-0.5 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-left pl-1',
                        selectedStage === stage.id && selectedStatuses.includes('running') && 'bg-blue-50 dark:bg-blue-950/20'
                      )}
                    >
                      <span className="text-[9px] text-zinc-400 uppercase font-medium">Run</span>
                      <span className={cn('text-xs font-bold', runningCount > 0 ? 'text-blue-500' : 'text-zinc-400')}>
                        {runningCount}
                      </span>
                    </button>

                    {/* Paused */}
                    <button
                      onClick={() => handleStageStatusClick(stage.id, 'paused')}
                      className={cn(
                        'flex flex-col p-0.5 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-left pl-1',
                        selectedStage === stage.id && selectedStatuses.includes('paused') && 'bg-amber-50 dark:bg-amber-950/20'
                      )}
                    >
                      <span className="text-[9px] text-zinc-400 uppercase font-medium">Pause</span>
                      <span className={cn('text-xs font-bold', pausedCount > 0 ? 'text-amber-500' : 'text-zinc-400')}>
                        {pausedCount}
                      </span>
                    </button>

                    {/* Done */}
                    <button
                      onClick={() => handleStageStatusClick(stage.id, 'done')}
                      className={cn(
                        'flex flex-col p-0.5 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-left pl-1',
                        selectedStage === stage.id && selectedStatuses.includes('done') && 'bg-emerald-50 dark:bg-emerald-950/20'
                      )}
                    >
                      <span className="text-[9px] text-zinc-400 uppercase font-medium">Done</span>
                      <span className={cn('text-xs font-bold', doneCount > 0 ? 'text-emerald-500' : 'text-zinc-400')}>
                        {doneCount}
                      </span>
                    </button>

                    {/* Failed */}
                    <button
                      onClick={() => handleStageStatusClick(stage.id, 'failed')}
                      className={cn(
                        'flex flex-col p-0.5 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-left pl-1',
                        selectedStage === stage.id && selectedStatuses.includes('failed') && 'bg-red-50 dark:bg-red-950/20'
                      )}
                    >
                      <span className="text-[9px] text-zinc-400 uppercase font-medium">Fail</span>
                      <span className={cn('text-xs font-bold', failedCount > 0 ? 'text-red-500 font-extrabold' : 'text-zinc-400')}>
                        {failedCount}
                      </span>
                    </button>
                  </div>

                  {/* Speed metrics */}
                  {(docsPerMin > 0 || (avgDur !== null && avgDur !== undefined)) && (
                    <div className="flex items-center gap-2 text-[10px] text-zinc-400 border-t pt-1 mt-0.5">
                      <ZapIcon className="size-3 text-amber-400 shrink-0" />
                      {docsPerMin > 0 && (
                        <span className="text-zinc-600 dark:text-zinc-300 font-medium">{docsPerMin}/min</span>
                      )}
                      {avgDur !== null && avgDur !== undefined && (
                        <span className="text-zinc-500">~{avgDur < 60 ? `${avgDur.toFixed(1)}s` : `${(avgDur / 60).toFixed(1)}m`}</span>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Filter and Table Toolbar */}
      <div className="flex flex-col gap-4 bg-zinc-50/50 dark:bg-zinc-900/30 p-4 rounded-xl border border-zinc-100 dark:border-zinc-800/80">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative w-64">
              <SearchIcon className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-400" />
              <input
                type="text"
                placeholder={t('monitorPanel.searchPlaceholder', 'Search document name or ID...')}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full h-9 pl-9 pr-4 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-background text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* Workspace Selector */}
            <select
              value={activeWorkspaceId}
              onChange={(e) => {
                setActiveWorkspaceId(e.target.value)
              }}
              className="h-9 px-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-background text-sm font-medium focus:outline-none"
            >
              <option value="all">All Workspaces</option>
              {workspaces.map((w) => (
                <option key={w.workspace_id} value={w.workspace_id}>
                  {w.workspace_name === w.workspace_id ? w.workspace_id : `${w.workspace_name} (${w.workspace_id})`}
                </option>
              ))}
            </select>

            {/* Stage Selector */}
            <select
              value={selectedStage || ''}
              onChange={(e) => {
                setSelectedStage(e.target.value || null)
                setPage(1)
              }}
              className="h-9 px-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-background text-sm focus:outline-none"
            >
              <option value="">{t('monitorPanel.allStages', 'All Stages')}</option>
              {CORE_STAGES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>

            {/* Status Multi-Select */}
            <div className="flex items-center gap-1.5">
              {STATUS_OPTIONS.map((s) => {
                const isActive = selectedStatuses.includes(s.id)
                return (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSelectedStatuses((prev) =>
                        prev.includes(s.id)
                          ? prev.filter((x) => x !== s.id)
                          : [...prev, s.id]
                      )
                      setPage(1)
                    }}
                    className={cn(
                      'h-9 px-3 rounded-lg border text-sm font-medium transition-all duration-150',
                      isActive
                        ? s.id === 'running'
                          ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-800 text-blue-700 dark:text-blue-400'
                          : s.id === 'paused'
                            ? 'bg-amber-50 dark:bg-amber-950/30 border-amber-300 dark:border-amber-800 text-amber-700 dark:text-amber-400'
                            : s.id === 'done'
                              ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400'
                              : 'bg-red-50 dark:bg-red-950/30 border-red-300 dark:border-red-800 text-red-700 dark:text-red-400'
                        : 'border-zinc-200 dark:border-zinc-800 bg-background text-zinc-500 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-900'
                    )}
                  >
                    {s.label}
                  </button>
                )
              })}
            </div>

            {(selectedStage || selectedStatuses.length > 0 || search || activeDocType !== 'all') && (
              <Button variant="ghost" size="sm" onClick={resetFilters} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">
                <XIcon className="size-4 mr-1.5" />
                Clear Filters
              </Button>
            )}
          </div>

          {/* Bulk Actions */}
          <div className="flex items-center gap-2">
            {selectedDocIds.length > 0 && (
              <span className="text-xs font-semibold text-zinc-500 mr-2">
                {selectedDocIds.length} Selected
              </span>
            )}
            <Button
              variant="default"
              size="sm"
              onClick={handleReprocessSelected}
              disabled={selectedDocIds.length === 0}
              className="bg-emerald-500 hover:bg-emerald-600 text-white font-medium shadow-sm transition-colors"
            >
              <PlayIcon className="size-4 mr-2" />
              {t('monitorPanel.reprocessSelected', 'Reprocess Selected')}
            </Button>
          </div>
        </div>
      </div>

      {/* Main Table */}
      {/* FIXED: Thêm min-h-[400px] để tránh bảng bị thu gọn về 0px khi panel config mở ra */}
      <div className="grow min-h-[400px] overflow-hidden flex flex-col border rounded-xl bg-background">
        <div className="grow overflow-auto">
          <Table className="border-collapse">
            <TableHeader className="sticky top-0 bg-background z-10 shadow-sm border-b">
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-12 text-center">
                  <Checkbox
                    checked={isAllPageSelected || (isSomePageSelected ? 'indeterminate' : false)}
                    onCheckedChange={handleSelectAll}
                  />
                </TableHead>
                <TableHead>{t('monitorPanel.document', 'Document')}</TableHead>
                <TableHead className="w-28">{t('monitorPanel.docType', 'Type')}</TableHead>
                <TableHead>Workspace</TableHead>
                <TableHead>{t('monitorPanel.stage', 'Stage')}</TableHead>
                <TableHead className="w-24 text-center">Shard</TableHead>
                <TableHead className="w-32">{t('monitorPanel.status', 'Status')}</TableHead>
                <TableHead className="w-24 text-center">{t('monitorPanel.attempts', 'Attempts')}</TableHead>
                <TableHead className="w-48">{t('monitorPanel.started', 'Started')}</TableHead>
                <TableHead className="w-32">{t('monitorPanel.duration', 'Duration')}</TableHead>
                <TableHead className="w-32 text-right">{t('monitorPanel.actions', 'Actions')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={11} className="h-48 text-center">
                    <div className="flex flex-col items-center justify-center gap-2 text-zinc-500">
                      <RefreshCwIcon className="h-8 w-8 animate-spin text-emerald-500" />
                      <span>Loading runs...</span>
                    </div>
                  </TableCell>
                </TableRow>
              ) : runs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={11} className="h-48 text-center">
                    <div className="flex flex-col items-center justify-center p-6">
                      <FileTextIcon className="h-10 w-10 text-zinc-300 dark:text-zinc-700 mb-2" />
                      <p className="text-sm text-zinc-500 dark:text-zinc-400">
                        {t('monitorPanel.noStageRuns', 'No stage runs matching current filters.')}
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                runs.map((run) => {
                  const hasError = !!run.error_json
                  const isSelected = selectedDocIds.includes(run.doc_id)

                  return (
                    <TableRow
                      key={`${run.run_id}-${run.stage}-${run.shard_index}`}
                      className={cn(
                        isSelected && 'bg-zinc-50/50 dark:bg-zinc-900/20'
                      )}
                    >
                      <TableCell className="text-center">
                        <Checkbox
                          checked={isSelected}
                          onCheckedChange={(checked) => handleSelectRow(run.doc_id, checked === true)}
                        />
                      </TableCell>
                      <TableCell className="max-w-md font-medium text-zinc-900 dark:text-zinc-100">
                        <div className="flex flex-col gap-0.5 truncate">
                          <span className="truncate" title={run.document_name || run.doc_id}>
                            {run.document_name || run.doc_id}
                          </span>
                          {run.document_name && (
                            <span className="text-[10px] text-zinc-400 dark:text-zinc-500 font-mono select-all">
                              {run.doc_id}
                            </span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        {run.doc_type === 'legal' ? (
                          <Badge
                            variant="outline"
                            className="bg-purple-50 dark:bg-purple-950/30 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800/60 font-medium text-[11px] px-2 py-0.5 inline-flex items-center gap-1"
                          >
                            <ScaleIcon className="size-3 text-purple-500" />
                            {t('monitorPanel.badges.legal', 'Legal')}
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800/60 font-medium text-[11px] px-2 py-0.5 inline-flex items-center gap-1"
                          >
                            <FileTextIcon className="size-3 text-blue-500" />
                            {t('monitorPanel.badges.general', 'General')}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <span className="text-xs bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 px-2 py-1 rounded font-semibold font-mono">
                          {run.workspace_id}
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="text-xs bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 px-2 py-1 rounded font-semibold font-mono">
                          {run.stage}
                        </span>
                      </TableCell>
                      <TableCell className="text-center text-xs text-zinc-600 dark:text-zinc-400 font-mono">
                        {run.shard_index + 1}/{run.shard_count}
                      </TableCell>
                      <TableCell>
                        <span
                          className={cn(
                            'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold select-none border',
                            run.status === 'done' && 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/30',
                            run.status === 'running' && 'bg-blue-50 dark:bg-blue-950/20 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-900/30',
                            run.status === 'paused' && 'bg-amber-50 dark:bg-amber-950/20 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-900/30',
                            run.status === 'failed' && 'bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900/30'
                          )}
                        >
                          <span
                            className={cn(
                              'size-1.5 rounded-full',
                              run.status === 'done' && 'bg-emerald-500',
                              run.status === 'running' && 'bg-blue-500 animate-pulse',
                              run.status === 'paused' && 'bg-amber-500',
                              run.status === 'failed' && 'bg-red-500'
                            )}
                          />
                          {run.status === 'done'
                            ? 'Done'
                            : run.status === 'running'
                              ? 'Running'
                              : run.status === 'paused'
                                ? t('monitorPanel.paused', 'Paused')
                                : 'Failed'}
                        </span>
                      </TableCell>
                      <TableCell className="text-center font-semibold font-mono text-zinc-700 dark:text-zinc-300">
                        {run.attempt}
                      </TableCell>
                      <TableCell className="text-xs text-zinc-500 font-mono">
                        {run.started_at ? new Date(run.started_at).toLocaleString() : '-'}
                      </TableCell>
                      <TableCell className="text-xs text-zinc-500 font-mono">
                        {getDurationText(run.started_at, run.completed_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          {run.status === 'paused' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2.5 text-amber-600 hover:text-emerald-600 dark:text-amber-400 dark:hover:text-emerald-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/80"
                              onClick={() => handleApproveStage(run.run_id, run.stage, run.shard_index, run.workspace_id)}
                              tooltip={t('monitorPanel.approve', 'Approve / Resume')}
                            >
                              <PlayIcon className="size-4" />
                            </Button>
                          )}
                          {run.status === 'failed' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2.5 text-red-600 hover:text-emerald-600 dark:text-red-400 dark:hover:text-emerald-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/80"
                              onClick={() => handleReplayStage(run)}
                              tooltip={t('monitorPanel.replayStage', 'Replay failed stage')}
                            >
                              <RotateCcwIcon className="size-4" />
                            </Button>
                          )}
                          {hasError && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2.5 text-zinc-500 hover:text-red-600 dark:hover:text-red-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/80"
                              onClick={() => {
                                setActiveError({
                                  docId: run.doc_id,
                                  documentName: run.document_name,
                                  stage: run.stage,
                                  shard: `${run.shard_index + 1}/${run.shard_count}`,
                                  error: run.error_json
                                })
                                setErrorModalOpen(true)
                              }}
                              tooltip={t('monitorPanel.viewError', 'View Error')}
                            >
                              <AlertTriangleIcon className="size-4" />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 px-2.5 text-zinc-500 hover:text-emerald-600 dark:hover:text-emerald-400 hover:bg-zinc-100 dark:hover:bg-zinc-800/80"
                            onClick={() => handleReprocessSingle(run.doc_id)}
                            tooltip={t('monitorPanel.reprocess', 'Reprocess')}
                          >
                            <RotateCcwIcon className="size-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                })
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination Footer */}
        {!loading && runs.length > 0 && (
          <div className="border-t p-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <span>{t('pagination.pageSize', 'Page Size')}:</span>
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value))
                  setPage(1)
                }}
                className="border rounded px-1.5 py-0.5 bg-background focus:outline-none"
              >
                {[10, 20, 50, 100].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>

            <PaginationControls
              currentPage={page}
              pageSize={pageSize}
              totalCount={total}
              totalPages={Math.ceil(total / pageSize)}
              onPageChange={setPage}
              onPageSizeChange={(size) => {
                setPageSize(size)
                setPage(1)
              }}
            />
          </div>
        )}
      </div>

      {/* Error Traceback Modal */}
      <Dialog open={errorModalOpen} onOpenChange={setErrorModalOpen}>
        <DialogContent className="max-w-2xl bg-background border rounded-xl shadow-2xl p-6">
          <DialogHeader className="border-b pb-3 mb-4 flex items-center justify-between">
            <DialogTitle className="text-lg font-bold text-zinc-900 dark:text-zinc-50 flex items-center gap-2">
              <AlertTriangleIcon className="size-5 text-red-500" />
              {t('monitorPanel.errorDetail', 'Error Detail')}
            </DialogTitle>
          </DialogHeader>

          {activeError && (
            <div className="flex flex-col gap-4 text-sm text-zinc-600 dark:text-zinc-400">
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 bg-zinc-50 dark:bg-zinc-900/50 p-3 rounded-lg border border-zinc-100 dark:border-zinc-800/80">
                <div>
                  <span className="text-xs text-zinc-400 uppercase font-semibold">Document ID</span>
                  <div className="font-mono text-xs select-all text-zinc-800 dark:text-zinc-200 truncate mt-0.5" title={activeError.docId}>
                    {activeError.docId}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-zinc-400 uppercase font-semibold">Document Name</span>
                  <div className="text-zinc-800 dark:text-zinc-200 truncate mt-0.5" title={activeError.documentName || '-'}>
                    {activeError.documentName || '-'}
                  </div>
                </div>
                <div className="mt-2">
                  <span className="text-xs text-zinc-400 uppercase font-semibold">Stage</span>
                  <div className="mt-0.5">
                    <span className="text-xs bg-zinc-200/60 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 px-2 py-0.5 rounded font-mono">
                      {activeError.stage}
                    </span>
                  </div>
                </div>
                <div className="mt-2">
                  <span className="text-xs text-zinc-400 uppercase font-semibold">Shard</span>
                  <div className="font-mono text-zinc-800 dark:text-zinc-200 mt-0.5">
                    {activeError.shard}
                  </div>
                </div>
              </div>

              <div className="flex flex-col gap-1.5 mt-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500 font-semibold uppercase">Traceback / Message</span>
                  <Button variant="ghost" size="sm" className="h-7 px-2 text-zinc-500 hover:text-zinc-800" onClick={copyTrace}>
                    {copied ? (
                      <>
                        <CheckIcon className="size-3.5 mr-1.5 text-emerald-500" />
                        Copied
                      </>
                    ) : (
                      <>
                        <CopyIcon className="size-3.5 mr-1.5" />
                        Copy
                      </>
                    )}
                  </Button>
                </div>
                <pre className="bg-zinc-100 dark:bg-zinc-800 p-4 rounded-lg overflow-auto text-xs max-h-[350px] whitespace-pre-wrap font-mono border border-zinc-200/60 dark:border-zinc-700/80 text-zinc-800 dark:text-zinc-200 shadow-inner">
                  {activeError.error?.traceback || activeError.error?.message || JSON.stringify(activeError.error, null, 2)}
                </pre>
              </div>

              <div className="flex justify-end gap-2 border-t pt-4 mt-2">
                <DialogClose asChild>
                  <Button variant="outline" size="sm">
                    {t('monitorPanel.close', 'Close')}
                  </Button>
                </DialogClose>
                <Button
                  variant="default"
                  size="sm"
                  className="bg-emerald-500 hover:bg-emerald-600 text-white"
                  onClick={() => {
                    setErrorModalOpen(false)
                    handleReprocessSingle(activeError.docId)
                  }}
                >
                  <RotateCcwIcon className="size-4 mr-2" />
                  {t('monitorPanel.reprocess', 'Reprocess')}
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}