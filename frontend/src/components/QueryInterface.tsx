'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Search, Send, Sparkles, Clock, FileText, Users, TrendingUp } from 'lucide-react'

interface QueryResult {
  answer: string
  confidence: number
  reasoning_path: string[]
  evidence: Array<{
    source: string
    content: string
    confidence: number
  }>
  entities_mentioned: string[]
  processingTime: number
}

interface QuerySuggestion {
  text: string
  type: 'entity' | 'relationship' | 'reasoning' | 'temporal'
  popularity: number
}

interface QueryInterfaceProps {
  onQuery: (query: string) => Promise<QueryResult>
  suggestions?: QuerySuggestion[]
  className?: string
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({
  onQuery,
  suggestions = [],
  className = ''
}) => {
  const [query, setQuery] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [filteredSuggestions, setFilteredSuggestions] = useState<QuerySuggestion[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [recentQueries, setRecentQueries] = useState<string[]>([])
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(-1)
  
  const inputRef = useRef<HTMLInputElement>(null)
  const suggestionsRef = useRef<HTMLDivElement>(null)

  // Filter suggestions based on query
  useEffect(() => {
    if (!query) {
      setFilteredSuggestions(suggestions.slice(0, 5))
    } else {
      const filtered = suggestions
        .filter(s => s.text.toLowerCase().includes(query.toLowerCase()))
        .sort((a, b) => b.popularity - a.popularity)
        .slice(0, 5)
      setFilteredSuggestions(filtered)
    }
    setActiveSuggestionIndex(-1)
  }, [query, suggestions])

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!showSuggestions) return
      
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveSuggestionIndex(prev => 
          prev < filteredSuggestions.length - 1 ? prev + 1 : 0
        )
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveSuggestionIndex(prev => prev > 0 ? prev - 1 : filteredSuggestions.length - 1)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (activeSuggestionIndex >= 0) {
          selectSuggestion(filteredSuggestions[activeSuggestionIndex])
        } else {
          handleQuery()
        }
      } else if (e.key === 'Escape') {
        setShowSuggestions(false)
        setActiveSuggestionIndex(-1)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [showSuggestions, filteredSuggestions, activeSuggestionIndex])

  // Click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleQuery = async () => {
    if (!query.trim() || isProcessing) return

    setIsProcessing(true)
    setResult(null)
    setShowSuggestions(false)

    try {
      const startTime = Date.now()
      const queryResult = await onQuery(query.trim())
      const processingTime = (Date.now() - startTime) / 1000

      setResult({
        ...queryResult,
        processingTime
      })

      // Add to recent queries
      setRecentQueries(prev => [query.trim(), ...prev.slice(0, 9)])
    } catch (error) {
      console.error('Query failed:', error)
      setResult({
        answer: 'Sorry, I encountered an error processing your query. Please try again.',
        confidence: 0,
        reasoning_path: [],
        evidence: [],
        entities_mentioned: [],
        processingTime: 0
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const selectSuggestion = (suggestion: QuerySuggestion) => {
    setQuery(suggestion.text)
    setShowSuggestions(false)
    setActiveSuggestionIndex(-1)
    inputRef.current?.focus()
  }

  const getSuggestionIcon = (type: string) => {
    switch (type) {
      case 'entity':
        return <Users className="w-4 h-4" />
      case 'relationship':
        return <TrendingUp className="w-4 h-4" />
      case 'temporal':
        return <Clock className="w-4 h-4" />
      default:
        return <Sparkles className="w-4 h-4" />
    }
  }

  const getSuggestionColor = (type: string) => {
    switch (type) {
      case 'entity':
        return 'text-blue-600'
      case 'relationship':
        return 'text-green-600'
      case 'temporal':
        return 'text-purple-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <div className={`bg-white rounded-xl shadow-lg border border-gray-200 ${className}`}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Query REXI</h2>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <Search className="w-4 h-4" />
            <span>Natural Language Processing</span>
          </div>
        </div>

        {/* Query Input */}
        <div className="relative mb-6">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setShowSuggestions(true)}
              placeholder="Ask me anything about your knowledge graph..."
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              disabled={isProcessing}
            />
            <button
              onClick={handleQuery}
              disabled={!query.trim() || isProcessing}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
            >
              {isProcessing ? (
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Suggestions Dropdown */}
          {showSuggestions && filteredSuggestions.length > 0 && (
            <div
              ref={suggestionsRef}
              className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10"
            >
              {filteredSuggestions.map((suggestion, index) => (
                <div
                  key={suggestion.text}
                  className={`px-4 py-3 hover:bg-gray-50 cursor-pointer flex items-center justify-between transition-colors duration-150 ${
                    index === activeSuggestionIndex ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => selectSuggestion(suggestion)}
                >
                  <div className="flex items-center space-x-3">
                    <div className={getSuggestionColor(suggestion.type)}>
                      {getSuggestionIcon(suggestion.type)}
                    </div>
                    <span className="text-gray-900">{suggestion.text}</span>
                  </div>
                  <div className="text-xs text-gray-500 capitalize">{suggestion.type}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Queries */}
        {recentQueries.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Recent Queries</h3>
            <div className="flex flex-wrap gap-2">
              {recentQueries.slice(0, 5).map((recentQuery, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(recentQuery)}
                  className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm hover:bg-gray-200 transition-colors duration-150"
                >
                  {recentQuery}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full mr-3"></div>
            <span className="text-gray-600">Processing your query...</span>
          </div>
        )}

        {/* Results */}
        {result && !isProcessing && (
          <div className="space-y-6">
            {/* Main Answer */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900">Answer</h3>
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <span>Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                  <span>Time: {result.processingTime.toFixed(2)}s</span>
                </div>
              </div>
              <p className="text-gray-700 leading-relaxed">{result.answer}</p>
            </div>

            {/* Reasoning Path */}
            {result.reasoning_path.length > 0 && (
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Reasoning Path</h3>
                <ol className="space-y-2">
                  {result.reasoning_path.map((step, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <span className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-medium">
                        {index + 1}
                      </span>
                      <span className="text-gray-700 text-sm">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* Evidence */}
            {result.evidence.length > 0 && (
              <div className="bg-green-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Evidence</h3>
                <div className="space-y-3">
                  {result.evidence.map((evidence, index) => (
                    <div key={index} className="border-l-4 border-green-400 pl-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-gray-900">{evidence.source}</span>
                        <span className="text-xs text-gray-500">
                          {(evidence.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                      <p className="text-sm text-gray-600">{evidence.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Entities Mentioned */}
            {result.entities_mentioned.length > 0 && (
              <div className="bg-purple-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Entities Mentioned</h3>
                <div className="flex flex-wrap gap-2">
                  {result.entities_mentioned.map((entity, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm"
                    >
                      {entity}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default QueryInterface
