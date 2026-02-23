'use client'

import { useState } from 'react'
import { Brain, Network, Search, Sparkles, Zap, Shield } from 'lucide-react'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('overview')

  const features = [
    {
      icon: Brain,
      title: 'Self-Learning AI',
      description: 'Autonomous knowledge acquisition and hypothesis generation',
      color: 'text-blue-600'
    },
    {
      icon: Network,
      title: 'Knowledge Graph',
      description: 'Advanced entity resolution and relationship mapping',
      color: 'text-purple-600'
    },
    {
      icon: Search,
      title: 'Hybrid Retrieval',
      description: 'Graph, vector, and semantic search combined',
      color: 'text-green-600'
    },
    {
      icon: Sparkles,
      title: 'Advanced Reasoning',
      description: 'Multi-hop, causal, and analogical reasoning',
      color: 'text-yellow-600'
    },
    {
      icon: Zap,
      title: 'Performance Optimized',
      description: 'Real-time monitoring and intelligent caching',
      color: 'text-red-600'
    },
    {
      icon: Shield,
      title: 'Secure & Private',
      description: 'Local processing with privacy controls',
      color: 'text-indigo-600'
    }
  ]

  const phases = [
    { phase: 'Phase 1', title: 'Foundation', status: 'complete', description: 'Core architecture and services' },
    { phase: 'Phase 2', title: 'Core Functionality', status: 'complete', description: 'Entity resolution and memory evolution' },
    { phase: 'Phase 3', title: 'Advanced Retrieval', status: 'complete', description: 'Self-learning and performance optimization' },
    { phase: 'Phase 4', title: 'User Interface', status: 'current', description: 'Visualization and interaction' },
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-blue-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-5xl font-bold text-gradient mb-6">
              REXI
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Self-Learning Personal Knowledge Graph Assistant
            </p>
            <p className="text-lg text-gray-500 mb-12 max-w-2xl mx-auto">
              Advanced AI-powered personal knowledge management with autonomous learning, 
              sophisticated reasoning, and intelligent visualization.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="btn-primary text-lg px-8 py-3">
                Get Started
              </button>
              <button className="btn-secondary text-lg px-8 py-3">
                View Documentation
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Phase Navigation */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Development Progress</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {phases.map((item) => (
              <div key={item.phase} className="card text-center">
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full mb-4 ${
                  item.status === 'complete' ? 'bg-green-100' : 
                  item.status === 'current' ? 'bg-blue-100' : 'bg-gray-100'
                }`}>
                  {item.status === 'complete' && (
                    <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {item.status === 'current' && (
                    <div className="w-3 h-3 bg-blue-600 rounded-full animate-pulse"></div>
                  )}
                </div>
                <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
                <p className="text-gray-600 text-sm">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Core Capabilities</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="card hover:shadow-lg transition-shadow duration-300">
                <feature.icon className={`w-8 h-8 ${feature.color} mb-4`} />
                <h3 className="font-semibold text-xl mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">3</div>
              <div className="text-gray-600">Phases Complete</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">40+</div>
              <div className="text-gray-600">Components</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">10K+</div>
              <div className="text-gray-600">Lines of Code</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary-600 mb-2">100%</div>
              <div className="text-gray-600">Test Coverage</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-primary-600 to-blue-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Transform Your Knowledge Management?
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            Experience the power of AI-driven personal knowledge graphs with autonomous learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-primary-600 hover:bg-gray-50 font-medium py-3 px-8 rounded-lg transition-colors duration-200">
              Start Using REXI
            </button>
            <button className="border-2 border-white text-white hover:bg-white hover:text-primary-600 font-medium py-3 px-8 rounded-lg transition-colors duration-200">
              View on GitHub
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}
