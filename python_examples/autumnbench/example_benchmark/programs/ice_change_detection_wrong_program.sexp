(program
      (= GRID_SIZE 16)
      (object CelestialBody (: day Bool) (list (Cell 0 0 (if day then "gold" else "gray"))
                                              (Cell 0 1 (if day then "gold" else "gray"))
                                              (Cell 1 0 (if day then "gold" else "gray"))
                                              (Cell 1 1 (if day then "gold" else "gray"))))
      (object Cloud (list (Cell -1 0 "gray")
                          (Cell 0 0 "gray")
                          (Cell 1 0 "gray")))

      (object Water (: liquid Bool) (Cell 0 0 (if liquid then "blue" else "lightblue")))

      (: celestialBody CelestialBody)
      (= celestialBody (initnext (CelestialBody true (Position 0 0)) (prev celestialBody)))

      (: cloud Cloud)
      (= cloud (initnext (Cloud (Position 4 0)) (prev cloud)))

      (: cnt Number)
      (= cnt (initnext 0 (prev cnt)))

      (: water (List Water))
      (= water (initnext (list) (updateObj (prev water) nextWater)))

      (on left (let
        (= cloud (nextCloud cloud (Position -1 0)))))
      (on right (let
        (= cloud (nextCloud cloud (Position 1 0)))))
      (on down (let
        (= water (if (< cnt 2) then
        (addObj water (Water (.. celestialBody day) (movePos (.. cloud origin) (Position 0 1))))
        else (addObj water (Water (! (.. celestialBody day)) (movePos (.. cloud origin) (Position 0 1))))))
        ))

      (on clicked (let
        (= celestialBody (updateObj celestialBody "day" (! (.. celestialBody day))))
        (= water (updateObj water (--> drop (updateObj drop "liquid" (! (.. drop liquid))))))
      ))

      (on clicked (= cnt (+ (prev cnt) 1)))

      (= nextWater (--> (drop)
                      (if (.. drop liquid)
                        then (nextLiquid drop)
                        else (nextSolid drop))))

      (= nextCloud (--> (cloud position)
                      (if (isWithinBounds (move cloud position))
                        then (move cloud position)
                        else cloud)))
  )