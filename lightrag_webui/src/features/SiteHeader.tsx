import { useEffect, useMemo, useState } from 'react'
import Button from '@/components/ui/Button'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import AppSettings from '@/components/AppSettings'
import { getWorkspaces, type WorkspaceItem } from '@/api/lightrag'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import { navigationService } from '@/services/navigation'
import { ZapIcon, GithubIcon, LogOutIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
  const { t } = useTranslation()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          {t('header.documents')}
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          {t('header.knowledgeGraph')}
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          {t('header.retrieval')}
        </NavigationTab>
        <NavigationTab value="api" currentTab={currentTab}>
          {t('header.api')}
        </NavigationTab>
      </TabsList>
    </div>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const { isGuestMode, coreVersion, apiVersion, username, webuiTitle, webuiDescription } = useAuthStore()
  const currentTab = useSettingsStore.use.currentTab()
  const selectedWorkspaceId = useSettingsStore.use.selectedWorkspaceId()
  const workspaceHistory = useSettingsStore.use.workspaceHistory()
  const setSelectedWorkspaceId = useSettingsStore.use.setSelectedWorkspaceId()
  const [workspaceRemoteOptions, setWorkspaceRemoteOptions] = useState<WorkspaceItem[]>([])

  useEffect(() => {
    let disposed = false

    const loadWorkspaces = async () => {
      try {
        const response = await getWorkspaces()
        if (disposed) return
        const apiWorkspaces = (response.workspace_items || [])
          .map((item) => ({
            workspace_id: item.workspace_id?.trim() || '',
            workspace_name: item.workspace_name?.trim() || item.workspace_id?.trim() || ''
          }))
          .filter((item) => !!item.workspace_id)
        setWorkspaceRemoteOptions(apiWorkspaces)

        const firstWorkspaceId = apiWorkspaces[0]?.workspace_id
        const selectedInRemoteList = apiWorkspaces.some(
          (item) => item.workspace_id === selectedWorkspaceId
        )

        // Default selection should be the first workspace in returned list.
        if (
          firstWorkspaceId &&
          (!selectedWorkspaceId ||
            selectedWorkspaceId === 'default' ||
            !selectedInRemoteList)
        ) {
          setSelectedWorkspaceId(firstWorkspaceId)
        }
      } catch (error) {
        console.error('Failed to load workspaces:', error)
      }
    }

    loadWorkspaces()
    return () => {
      disposed = true
    }
  }, [selectedWorkspaceId, setSelectedWorkspaceId])

  const workspaceOptions = useMemo(() => {
    const deduped = new Set<string>()
    const options: WorkspaceItem[] = []

    // Keep API order first so the first option can be used as default selection.
    for (const item of workspaceRemoteOptions) {
      const normalized = item.workspace_id.trim()
      if (!normalized || deduped.has(normalized)) continue
      deduped.add(normalized)
      options.push({
        workspace_id: normalized,
        workspace_name: item.workspace_name?.trim() || normalized
      })
    }

    for (const id of workspaceHistory) {
      const normalized = id.trim()
      if (!normalized || deduped.has(normalized)) continue
      deduped.add(normalized)
      options.push({ workspace_id: normalized, workspace_name: normalized })
    }

    if (options.length === 0) {
      const fallbackWorkspace = selectedWorkspaceId.trim() || 'default'
      options.push({ workspace_id: fallbackWorkspace, workspace_name: fallbackWorkspace })
    }

    return options
  }, [selectedWorkspaceId, workspaceHistory, workspaceRemoteOptions])

  const versionDisplay = (coreVersion && apiVersion)
    ? `${coreVersion}/${apiVersion}`
    : null;

  // Check if frontend needs rebuild (apiVersion ends with warning symbol)
  const hasWarning = apiVersion?.endsWith('⚠️');
  const versionTooltip = hasWarning
    ? t('header.frontendNeedsRebuild')
    : versionDisplay ? `v${versionDisplay}` : '';

  const handleLogout = () => {
    navigationService.navigateToLogin();
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="min-w-[200px] w-auto flex items-center">
        <a href={webuiPrefix} className="flex items-center gap-2">
          <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
        {webuiTitle && (
          <div className="flex items-center">
            <span className="mx-1 text-xs text-gray-500 dark:text-gray-400">|</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="font-medium text-sm cursor-default">
                    {webuiTitle}
                  </span>
                </TooltipTrigger>
                {webuiDescription && (
                  <TooltipContent side="bottom">
                    {webuiDescription}
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
        {isGuestMode && (
          <div className="ml-2 self-center px-2 py-1 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 rounded-md">
            {t('login.guestMode', 'Guest Mode')}
          </div>
        )}
      </div>

      <nav className="min-w-[420px] flex items-center justify-end">
        <div className="flex items-center gap-2">
          {(currentTab === 'documents' || currentTab === 'knowledge-graph' || currentTab === 'retrieval') && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400">Workspace</span>
              <select
                className="h-7 w-44 rounded-md border border-input bg-background px-2 text-xs"
                value={selectedWorkspaceId}
                onChange={(event) => setSelectedWorkspaceId(event.target.value)}
              >
                {workspaceOptions.map((item) => (
                  <option
                    key={item.workspace_id}
                    value={item.workspace_id}
                  >
                    {item.workspace_name === item.workspace_id
                      ? item.workspace_id
                      : `${item.workspace_name} (${item.workspace_id})`}
                  </option>
                ))}
              </select>
            </div>
          )}
          {versionDisplay && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1 cursor-default">
                    v{versionDisplay}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {versionTooltip}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <Button variant="ghost" size="icon" side="bottom" tooltip={t('header.projectRepository')}>
            <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
              <GithubIcon className="size-4" aria-hidden="true" />
            </a>
          </Button>
          <AppSettings />
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              side="bottom"
              tooltip={`${t('header.logout')} (${username})`}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </nav>
    </header>
  )
}
