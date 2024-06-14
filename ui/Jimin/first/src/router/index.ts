import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import HomeRoutes from '@/home/router/HomeRoutes'
import BoardRoutes from '@/board/router/BoardRoutes'
import ProductRoutes from '@/product/router/ProductRoutes'
import AccountRoutes from '@/account/router/AccountRoutes'
import AuthenticationRoutes from '@/authentication/router/AthenticationRoutes'

const routes: Array<RouteRecordRaw> = [
  ...HomeRoutes,
  // 실제 Javascript 던 Typescript 던 spread 문법을 가지고 있습니다.
  // 내부에 가지고 있는 내용을 그대로 분산시켜서 뒤에 붙이는 작업입니다.
  // 그러므로 ...BoardRoutes 는 BoardRoutes 내에 있는 모든 내용을 뒤에 붙입니다.
  ...BoardRoutes,
  ...ProductRoutes,
  ...AccountRoutes,
  ...AuthenticationRoutes
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router