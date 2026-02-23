'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Network, Zap, Eye, EyeOff, Download, RefreshCw } from 'lucide-react'

interface GraphNode {
  id: string
  name: string
  type: string
  confidence: number
  properties: Record<string, any>
  x?: number
  y?: number
  fx?: number
  fy?: number
}

interface GraphLink {
  source: string | GraphNode
  target: string | GraphNode
  type: string
  strength: number
  confidence: number
  properties: Record<string, any>
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

interface GraphVisualizationProps {
  data: GraphData
  width?: number
  height?: number
  onNodeClick?: (node: GraphNode) => void
  onLinkClick?: (link: GraphLink) => void
  className?: string
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  data,
  width = 800,
  height = 600,
  onNodeClick,
  onLinkClick,
  className = ''
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [selectedLink, setSelectedLink] = useState<GraphLink | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [isSimulationRunning, setIsSimulationRunning] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [filterType, setFilterType] = useState<string>('all')

  // Color scale for node types
  const colorScale = d3.scaleOrdinal<string>()
    .domain(['PERSON', 'CONCEPT', 'SKILL', 'TOPIC', 'PROJECT', 'TOOL', 'PAPER', 'BOOK', 'EVENT', 'ORGANIZATION'])
    .range(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'])

  // Filter data based on type
  const filteredData = React.useMemo(() => {
    if (filterType === 'all') return data
    
    const filteredNodeIds = new Set(
      data.nodes
        .filter(node => node.type === filterType)
        .map(node => node.id)
    )
    
    const filteredLinks = data.links.filter(
      link => 
        (typeof link.source === 'string' ? filteredNodeIds.has(link.source) : filteredNodeIds.has(link.source.id)) ||
        (typeof link.target === 'string' ? filteredNodeIds.has(link.target) : filteredNodeIds.has(link.target.id))
    )
    
    return {
      nodes: data.nodes.filter(node => filteredNodeIds.has(node.id)),
      links: filteredLinks
    }
  }, [data, filterType])

  useEffect(() => {
    if (!svgRef.current || !filteredData.nodes.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Create simulation
    const simulation = d3.forceSimulation(filteredData.nodes as any)
      .force('link', d3.forceLink(filteredData.links as any)
        .id((d: any) => d.id)
        .distance(100)
        .strength((d: any) => d.strength)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30))

    // Create container for zoom and pan
    const container = svg.append('g')

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Create links
    const link = container.append('g')
      .selectAll('line')
      .data(filteredData.links)
      .enter()
      .append('line')
      .attr('stroke', (d: any) => {
        const opacity = d.confidence || 0.5
        return `rgba(107, 114, 128, ${opacity})`
      })
      .attr('stroke-width', (d: any) => Math.max(1, (d.strength || 0.5) * 3))
      .attr('stroke-dasharray', (d: any) => {
        return d.type === 'CONTRADICTS' ? '5,5' : 'none'
      })
      .style('cursor', 'pointer')
      .on('click', (event, d: GraphLink) => {
        setSelectedLink(d)
        onLinkClick?.(d)
      })
      .on('mouseover', function(event, d: GraphLink) {
        d3.select(this).attr('stroke-width', 5)
      })
      .on('mouseout', function(event, d: GraphLink) {
        d3.select(this).attr('stroke-width', Math.max(1, (d.strength || 0.5) * 3))
      })

    // Create nodes
    const node = container.append('g')
      .selectAll('g')
      .data(filteredData.nodes)
      .enter()
      .append('g')
      .style('cursor', 'pointer')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d: GraphNode) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d: GraphNode) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d: GraphNode) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        })
      )
      .on('click', (event, d: GraphNode) => {
        setSelectedNode(d)
        onNodeClick?.(d)
      })
      .on('mouseover', function(event, d: GraphNode) {
        setHoveredNode(d)
        d3.select(this).select('circle').attr('r', 15)
      })
      .on('mouseout', function(event, d: GraphNode) {
        setHoveredNode(null)
        d3.select(this).select('circle').attr('r', 10)
      })

    // Add circles to nodes
    node.append('circle')
      .attr('r', 10)
      .attr('fill', (d: GraphNode) => colorScale(d.type) as string)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)

    // Add labels to nodes
    const labels = node.append('text')
      .text((d: GraphNode) => d.name)
      .attr('font-size', '12px')
      .attr('font-family', 'Inter, sans-serif')
      .attr('text-anchor', 'middle')
      .attr('dy', -15)
      .style('pointer-events', 'none')
      .style('display', showLabels ? 'block' : 'none')

    // Add confidence indicators
    node.append('circle')
      .attr('r', 3)
      .attr('cx', 8)
      .attr('cy', -8)
      .attr('fill', (d: GraphNode) => d.confidence > 0.8 ? '#10b981' : d.confidence > 0.5 ? '#f59e0b' : '#ef4444')
      .style('pointer-events', 'none')

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      node.attr('transform', (d: GraphNode) => `translate(${d.x},${d.y})`)
    })

    // Cleanup
    return () => {
      simulation.stop()
    }
  }, [filteredData, width, height, showLabels])

  // Toggle labels visibility
  useEffect(() => {
    if (!svgRef.current) return
    
    const labels = d3.select(svgRef.current).selectAll('text')
    labels.style('display', showLabels ? 'block' : 'none')
  }, [showLabels])

  // Toggle simulation
  const toggleSimulation = () => {
    setIsSimulationRunning(!isSimulationRunning)
    // In a real implementation, this would stop/start the simulation
  }

  // Export graph as SVG
  const exportGraph = () => {
    if (!svgRef.current) return
    
    const svgData = new XMLSerializer().serializeToString(svgRef.current)
    const blob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = 'rexi-graph.svg'
    link.click()
    
    URL.revokeObjectURL(url)
  }

  const nodeTypes = Array.from(new Set(data.nodes.map(n => n.type)))

  return (
    <div className={`bg-white rounded-xl shadow-lg border border-gray-200 ${className}`}>
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Knowledge Graph</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={toggleSimulation}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title={isSimulationRunning ? 'Pause Simulation' : 'Resume Simulation'}
            >
              {isSimulationRunning ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
            <button
              onClick={() => setShowLabels(!showLabels)}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title={showLabels ? 'Hide Labels' : 'Show Labels'}
            >
              <Network className="w-4 h-4" />
            </button>
            <button
              onClick={exportGraph}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Export Graph"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="mt-4 flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label htmlFor="filter-type" className="text-sm font-medium text-gray-700">Filter:</label>
            <select
              id="filter-type"
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Types</option>
              {nodeTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>{filteredData.nodes.length} nodes</span>
            <span>{filteredData.links.length} edges</span>
          </div>
        </div>
      </div>
      
      <div className="relative">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="w-full"
        />
        
        {/* Tooltip for hovered node */}
        {hoveredNode && (
          <div className="absolute top-4 left-4 bg-gray-900 text-white p-3 rounded-lg shadow-lg max-w-xs">
            <div className="font-semibold">{hoveredNode.name}</div>
            <div className="text-sm text-gray-300">{hoveredNode.type}</div>
            <div className="text-sm text-gray-300">Confidence: {(hoveredNode.confidence * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>
      
      {/* Selected node details */}
      {selectedNode && (
        <div className="absolute top-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 max-w-xs">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-gray-900">{selectedNode.name}</h4>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
          <div className="space-y-1 text-sm">
            <div><span className="font-medium">Type:</span> {selectedNode.type}</div>
            <div><span className="font-medium">Confidence:</span> {(selectedNode.confidence * 100).toFixed(1)}%</div>
            {selectedNode.properties.description && (
              <div><span className="font-medium">Description:</span> {selectedNode.properties.description}</div>
            )}
          </div>
        </div>
      )}
      
      {/* Selected link details */}
      {selectedLink && (
        <div className="absolute bottom-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 max-w-xs">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-gray-900">Relationship</h4>
            <button
              onClick={() => setSelectedLink(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
          <div className="space-y-1 text-sm">
            <div><span className="font-medium">Type:</span> {selectedLink.type}</div>
            <div><span className="font-medium">Strength:</span> {(selectedLink.strength * 100).toFixed(1)}%</div>
            <div><span className="font-medium">Confidence:</span> {(selectedLink.confidence * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default GraphVisualization
