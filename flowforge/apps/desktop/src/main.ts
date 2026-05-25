import { createApp } from 'vue'
import { createPinia } from 'pinia'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import App from './App.vue'
import { initializeStores } from './stores/initializeStores'
import { registerTrinityNodes } from './litegraph/nodes'

// Import global styles
import 'primeicons/primeicons.css'
import './styles/main.css'

// Create Vue app and Pinia store
const app = createApp(App)
const pinia = createPinia()

// Install Pinia
app.use(pinia)

// Install PrimeVue
app.use(PrimeVue, {
  unstyled: false, // PrimeVue provides structural CSS for Splitter layout
})
app.directive('tooltip', Tooltip)

// Initialize stores after pinia is set up
initializeStores()

// Register Trinity node types with LiteGraph
registerTrinityNodes()

// Mount the app
app.mount('#app')
